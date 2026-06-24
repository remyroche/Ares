#!/usr/bin/env python3
"""Anchored reliability-correction meta experiment.

This experiment keeps the production contract unchanged:

* unchanged ``y_bin`` label;
* one final meta probability;
* current OOF meta score as the frozen anchor ``meta_0``;
* reliability/context signals enter only through a constrained logit correction.

The main test is whether high-confidence failure, difficult-period, leaf, and
canonical context signals help when they are forced into ``z_final = z0 + delta``
instead of being diluted inside the full meta feature stack.
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


HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")

TRIAL_ANCHOR = "M0_anchor_current_meta"
TRIAL_R1 = "R1_z0_minus_q_fail_inc"
TRIAL_R2 = "R2_z0_minus_q_fail_inc_minus_q_period_inc"
TRIAL_R3 = "R3_period_conditioned_failure_penalty"
TRIAL_META_MAIN = "M1_shallow_clipped_main_effects"
TRIAL_META_INTERACTIONS = "M2_shallow_clipped_limited_interactions"
TRIALS = (TRIAL_ANCHOR, TRIAL_R1, TRIAL_R2, TRIAL_R3, TRIAL_META_MAIN, TRIAL_META_INTERACTIONS)

DIRECTIONAL_SUMMARY_COLUMNS = [
    "head",
    "arm",
    "distillation_variant",
    "directional_timestamp_count",
    "directional_eligible_rows",
    "timestamp_weighted_hr_top10",
    "baseline_timestamp_weighted_hr_top10",
    "delta_timestamp_weighted_hr_top10",
    "timestamp_weighted_hr_top20",
    "baseline_timestamp_weighted_hr_top20",
    "delta_timestamp_weighted_hr_top20",
    "timestamp_weighted_hr_top30",
    "baseline_timestamp_weighted_hr_top30",
    "delta_timestamp_weighted_hr_top30",
    "trade_weighted_hr_top30",
    "baseline_trade_weighted_hr_top30",
    "ndcg_top30",
    "baseline_ndcg_top30",
    "delta_ndcg_top30",
    "average_precision_top30",
    "baseline_average_precision_top30",
    "delta_average_precision_top30",
    "pairwise_concordance_top30",
    "baseline_pairwise_concordance_top30",
    "delta_pairwise_concordance_top30",
    "top30_jaccard",
    "top30_entrant_hit_rate",
    "top30_removed_hit_rate",
    "net_correct_trades_gained",
    "worst_week_hr_top30",
    "q10_week_hr_top30",
    "normal_period_delta_hr_top30",
    "bad_period_delta_hr_top30",
]

DIRECTIONAL_TIMESTAMP_COLUMNS = [
    "head",
    "arm",
    "distillation_variant",
    "timestamp",
    "week",
    "eligible_rows",
    "hr_top10",
    "baseline_hr_top10",
    "delta_hr_top10",
    "hr_top20",
    "baseline_hr_top20",
    "delta_hr_top20",
    "hr_top30",
    "baseline_hr_top30",
    "delta_hr_top30",
    "ndcg_top30",
    "baseline_ndcg_top30",
    "delta_ndcg_top30",
    "top30_jaccard",
    "top30_entrant_hit_rate",
    "top30_removed_hit_rate",
    "net_correct_trades_gained",
]

PATH_CONTROL_TOKENS = (
    "oof_pred",
    "oof_p_move",
    "oof_meta_clf",
    "oof_base_clf",
    "oof_lgbm_prob",
    "oof_lgbm_raw_score",
    "oof_prob_",
    "oof_raw_score_",
    "oof_margin",
    "oof_entropy",
    "oof_variance",
    "oof_rank_pct",
    "oof_score_margin",
    "oof_rank_margin",
    "oof_score_",
    "oof_rank_",
    "oof_score_path",
    "oof_rank_path",
)

FORBIDDEN_TOKENS = tuple(stack.FORBIDDEN_FEATURE_TOKENS) + (
    "rank_bin_win_rate",
    "rank_bin_lift",
    "rank_bin_net_ret",
    "leaf_target",
    "leaf_hit_rate",
    "leaf_error",
)


def _json_default(value: Any) -> Any:
    return ctx._json_default(value)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sigmoid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(arr, -50.0, 50.0)))).astype(np.float32, copy=False)


def _safe_logit(values: np.ndarray) -> np.ndarray:
    return stack._logit(values).astype(np.float32, copy=False)


def _clip_delta(delta: np.ndarray, clip: float) -> np.ndarray:
    return np.clip(np.asarray(delta, dtype=np.float32), -float(clip), float(clip)).astype(np.float32, copy=False)


def _score_with_delta(anchor_score: np.ndarray, delta: np.ndarray, clip: float) -> np.ndarray:
    z0 = _safe_logit(anchor_score)
    return _sigmoid(z0 + _clip_delta(delta, clip))


def _safe_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(index=getattr(frame, "index", None))
    out = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return _downcast_numeric(out)


def _select_path_control_columns(raw: pd.DataFrame, max_cols: int) -> list[str]:
    candidates: list[str] = []
    for col in raw.columns:
        low = str(col).lower()
        if any(tok in low for tok in FORBIDDEN_TOKENS):
            continue
        if any(tok in low for tok in PATH_CONTROL_TOKENS):
            candidates.append(str(col))
    scored: list[tuple[float, str]] = []
    for col in candidates:
        values = pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        if finite.mean() < 0.05:
            continue
        var = float(np.nanvar(values[finite])) if finite.any() else 0.0
        scored.append((finite.mean() * math.log1p(max(var, 0.0)), col))
    scored.sort(reverse=True)
    return [col for _score, col in scored[: int(max_cols)]]


def _anchor_control_features(
    raw: pd.DataFrame,
    timestamps: pd.Series,
    anchor_score: np.ndarray,
    *,
    max_path_cols: int,
    selected_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    score = np.clip(np.asarray(anchor_score, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    out = pd.DataFrame(
        {
            "anchor_p0": score,
            "anchor_logit0": _safe_logit(score),
            "anchor_rank0_by_timestamp": stack._rank_pct_by_timestamp(ts, score),
        }
    )
    cols = list(selected_cols) if selected_cols is not None else _select_path_control_columns(raw, int(max_path_cols))
    for col in cols:
        if col in raw.columns:
            out[f"control__{col}"] = pd.to_numeric(raw[col], errors="coerce").to_numpy(dtype=np.float32)
    return _downcast_numeric(out), cols


def _rank_features_from_values(timestamps: pd.Series, values: np.ndarray, prefix: str) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    vals = np.asarray(values, dtype=np.float32)
    frame = pd.DataFrame({"timestamp": ts, "value": vals})
    mean = frame.groupby("timestamp", sort=False)["value"].transform("mean").to_numpy(dtype=np.float32)
    return _downcast_numeric(
        pd.DataFrame(
            {
                prefix: vals,
                f"{prefix}_percentile": stack._rank_pct_by_timestamp(ts, vals),
                f"{prefix}_minus_timestamp_mean": (vals - mean).astype(np.float32, copy=False),
            }
        )
    )


def _constant_classifier_probs(
    train_len: int,
    valid_len: int,
    p: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    prob = float(np.clip(p if np.isfinite(p) else 0.5, 1e-4, 1.0 - 1e-4))
    return (
        np.full(train_len, prob, dtype=np.float32),
        np.full(valid_len, prob, dtype=np.float32),
        {"reason": "constant_insufficient_rows_or_classes", "feature_count": 0, "valid_auc": np.nan},
    )


def _fit_predict_failure_model(
    x_train: pd.DataFrame,
    target_train: np.ndarray,
    candidate_train: np.ndarray,
    x_valid: pd.DataFrame,
    *,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_ids = np.flatnonzero(candidate_train & np.isfinite(target_train))
    if train_ids.size < int(args.failure_min_train_rows) or len(np.unique(target_train[train_ids].astype(np.int8))) < 2:
        p = float(np.nanmean(target_train[train_ids])) if train_ids.size else 0.5
        return _constant_classifier_probs(len(x_train), len(x_valid), p)
    pred_valid, diag, model = stack._fit_basic_classifier(
        x_train.iloc[train_ids].reset_index(drop=True),
        target_train[train_ids].astype(np.int8),
        x_valid.reset_index(drop=True),
        np.zeros(len(x_valid), dtype=np.int8),
        seed=int(seed),
        max_depth=int(args.failure_max_depth),
        n_estimators=int(args.failure_n_estimators),
        min_child_fraction=float(args.failure_min_child_fraction),
    )
    if model is None:
        p = float(np.nanmean(target_train[train_ids]))
        pred_train = np.full(len(x_train), p, dtype=np.float32)
    else:
        keep_cols = [c for c in x_train.columns if pd.to_numeric(x_train[c], errors="coerce").notna().mean() > 0.02]
        x_all = pd.concat([x_train, x_train], axis=0, ignore_index=True, copy=False).replace([np.inf, -np.inf], np.nan)
        prepared = _prepare_model_matrix(x_all.loc[:, keep_cols])
        pred_train = model.predict_proba(prepared.iloc[: len(x_train)])[:, 1].astype(np.float32, copy=False)
    return pred_train, pred_valid.astype(np.float32, copy=False), diag


def _crossfit_failure_increment(
    *,
    controls_train: pd.DataFrame,
    full_train: pd.DataFrame,
    controls_valid: pd.DataFrame,
    full_valid: pd.DataFrame,
    train_timestamps: pd.Series,
    y_train: np.ndarray,
    anchor_train: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rank_train = stack._rank_pct_by_timestamp(train_timestamps, anchor_train)
    candidate = np.isfinite(rank_train) & (rank_train >= float(args.failure_candidate_rank_threshold)) & (np.asarray(y_train) >= 0)
    target = np.where(candidate, (np.asarray(y_train, dtype=np.int8) == 0).astype(np.int8), np.nan)
    q_ctrl_train = np.full(len(y_train), np.nan, dtype=np.float32)
    q_full_train = np.full(len(y_train), np.nan, dtype=np.float32)
    inner_valid = 0
    if len(y_train) >= 500 and int(args.inner_folds) >= 2:
        inner_folds = canon._make_chrono_folds(train_timestamps.reset_index(drop=True), int(args.inner_folds), embargo_hours=int(args.inner_embargo_hours))
        for inner in inner_folds:
            tr = np.asarray(inner.train_idx, dtype=np.int64)
            va = np.asarray(inner.valid_idx, dtype=np.int64)
            ctrl_tr, ctrl_va = controls_train.iloc[tr].reset_index(drop=True), controls_train.iloc[va].reset_index(drop=True)
            full_tr, full_va = full_train.iloc[tr].reset_index(drop=True), full_train.iloc[va].reset_index(drop=True)
            ctrl_pred_train, ctrl_pred_valid, _ = _fit_predict_failure_model(
                ctrl_tr,
                target[tr],
                candidate[tr],
                ctrl_va,
                seed=int(seed + 101 * inner.fold_id),
                args=args,
            )
            full_pred_train, full_pred_valid, _ = _fit_predict_failure_model(
                full_tr,
                target[tr],
                candidate[tr],
                full_va,
                seed=int(seed + 211 * inner.fold_id),
                args=args,
            )
            del ctrl_pred_train, full_pred_train
            q_ctrl_train[va] = ctrl_pred_valid
            q_full_train[va] = full_pred_valid
            inner_valid += int(np.isfinite(ctrl_pred_valid).sum())
    p_default = float(np.nanmean(target[candidate])) if candidate.any() else 0.5
    missing = ~np.isfinite(q_ctrl_train)
    q_ctrl_train[missing] = p_default
    q_full_train[~np.isfinite(q_full_train)] = p_default
    q_ctrl_fit_train, q_ctrl_valid, ctrl_diag = _fit_predict_failure_model(
        controls_train,
        target,
        candidate,
        controls_valid,
        seed=int(seed + 991),
        args=args,
    )
    q_full_fit_train, q_full_valid, full_diag = _fit_predict_failure_model(
        full_train,
        target,
        candidate,
        full_valid,
        seed=int(seed + 1991),
        args=args,
    )
    del q_ctrl_fit_train, q_full_fit_train
    inc_train = (_safe_logit(q_full_train) - _safe_logit(q_ctrl_train)).astype(np.float32, copy=False)
    inc_valid = (_safe_logit(q_full_valid) - _safe_logit(q_ctrl_valid)).astype(np.float32, copy=False)
    eval_mask = candidate & np.isfinite(q_ctrl_train) & np.isfinite(q_full_train)
    diag = {
        "failure_candidate_rank_threshold": float(args.failure_candidate_rank_threshold),
        "failure_candidate_rows": int(candidate.sum()),
        "failure_label_positive_count": int(np.nansum(target == 1)),
        "failure_label_rate": float(np.nanmean(target[candidate])) if candidate.any() else np.nan,
        "failure_inner_valid_rows": int(inner_valid),
        "failure_control_inner_auc": stack._safe_auc(target[eval_mask].astype(np.int8), q_ctrl_train[eval_mask]) if eval_mask.any() else np.nan,
        "failure_full_inner_auc": stack._safe_auc(target[eval_mask].astype(np.int8), q_full_train[eval_mask]) if eval_mask.any() else np.nan,
        "failure_increment_inner_auc_delta": np.nan,
        "failure_control_feature_count": int(ctrl_diag.get("feature_count", 0)),
        "failure_full_feature_count": int(full_diag.get("feature_count", 0)),
        "failure_control_reason": ctrl_diag.get("reason", ""),
        "failure_full_reason": full_diag.get("reason", ""),
    }
    if np.isfinite(diag["failure_control_inner_auc"]) and np.isfinite(diag["failure_full_inner_auc"]):
        diag["failure_increment_inner_auc_delta"] = float(diag["failure_full_inner_auc"] - diag["failure_control_inner_auc"])
    return inc_train, inc_valid, diag


def _timestamp_nuisance_table(timestamps: pd.Series, symbols: pd.Series | None = None) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    frame = pd.DataFrame({"timestamp": ts})
    if symbols is not None:
        frame["symbol"] = symbols.reset_index(drop=True).astype(str)
    grouped = frame.groupby("timestamp", sort=True)
    out = pd.DataFrame(index=pd.Index(sorted(ts.dropna().unique()), name="timestamp"))
    counts = grouped.size().reindex(out.index).astype(float)
    out["nuisance_log_row_count"] = np.log1p(counts).astype(np.float32)
    if "symbol" in frame.columns:
        out["nuisance_log_symbol_count"] = np.log1p(grouped["symbol"].nunique().reindex(out.index).astype(float)).astype(np.float32)
    hour = pd.Series(out.index).dt.hour.to_numpy(dtype=np.float64)
    dow = pd.Series(out.index).dt.dayofweek.to_numpy(dtype=np.float64)
    out["nuisance_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    out["nuisance_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    out["nuisance_dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32)
    out["nuisance_dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32)
    return _downcast_numeric(out)


def _predict_timestamp_classifier(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    pred_x: pd.DataFrame,
    *,
    seed: int,
    max_depth: int,
    n_estimators: int,
    min_child_fraction: float,
) -> tuple[np.ndarray, dict[str, Any], Any | None]:
    return stack._fit_basic_classifier(
        train_x.reset_index(drop=True),
        np.asarray(train_y, dtype=np.int8),
        pred_x.reset_index(drop=True),
        np.zeros(len(pred_x), dtype=np.int8),
        seed=int(seed),
        max_depth=int(max_depth),
        n_estimators=int(n_estimators),
        min_child_fraction=float(min_child_fraction),
    )


def _period_increment_features(
    *,
    z_train: pd.DataFrame,
    z_valid: pd.DataFrame,
    nuisance_train: pd.DataFrame,
    nuisance_valid: pd.DataFrame,
    train_timestamps: pd.Series,
    valid_timestamps: pd.Series,
    y_train: np.ndarray,
    anchor_train: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_hr = stack._baseline_timestamp_hr30(train_timestamps, y_train, anchor_train)
    labels = stack._difficult_period_labels(
        train_hr,
        short_window=int(args.period_short_window),
        long_window=int(args.period_long_window),
        quantile=float(args.period_difficult_quantile),
    )
    z = z_train.reindex(labels.index).copy()
    nui = nuisance_train.reindex(labels.index).copy()
    diag: dict[str, Any] = {
        "period_label_timestamps": int(len(labels)),
        "period_label_rate": float(labels.mean()) if len(labels) else np.nan,
    }
    if labels.empty or len(np.unique(labels.to_numpy(dtype=np.int8))) < 2 or z.dropna(how="all").shape[0] < int(args.period_min_train_timestamps):
        base = float(labels.mean()) if len(labels) else 0.5
        train_inc_ts = pd.Series(0.0, index=z_train.index, dtype="float32")
        valid_inc_ts = pd.Series(0.0, index=z_valid.index, dtype="float32")
        diag["period_reason"] = "constant_insufficient_period_labels"
    else:
        order = np.argsort(pd.to_datetime(labels.index, utc=True, errors="coerce").to_numpy(dtype="datetime64[ns]"), kind="mergesort")
        q_full_train = np.full(len(labels), np.nan, dtype=np.float32)
        q_nui_train = np.full(len(labels), np.nan, dtype=np.float32)
        if len(order) >= 120 and int(args.inner_folds) >= 2:
            split_points = np.linspace(0, len(order), int(args.inner_folds) + 1).round().astype(int)
            for i in range(int(args.inner_folds)):
                va_ids = order[split_points[i] : split_points[i + 1]]
                if len(va_ids) < 10:
                    continue
                tr_ids = order[: split_points[i]]
                if len(tr_ids) < 40:
                    tr_ids = np.setdiff1d(order, va_ids, assume_unique=False)
                if len(np.unique(labels.to_numpy(dtype=np.int8)[tr_ids])) < 2:
                    continue
                full_pred, _full_diag, _ = _predict_timestamp_classifier(
                    z.iloc[tr_ids],
                    labels.to_numpy(dtype=np.int8)[tr_ids],
                    z.iloc[va_ids],
                    seed=int(seed + 307 * (i + 1)),
                    max_depth=int(args.period_max_depth),
                    n_estimators=int(args.period_n_estimators),
                    min_child_fraction=float(args.period_min_child_fraction),
                )
                nui_pred, _nui_diag, _ = _predict_timestamp_classifier(
                    nui.iloc[tr_ids],
                    labels.to_numpy(dtype=np.int8)[tr_ids],
                    nui.iloc[va_ids],
                    seed=int(seed + 409 * (i + 1)),
                    max_depth=2,
                    n_estimators=max(80, int(args.period_n_estimators // 2)),
                    min_child_fraction=0.05,
                )
                q_full_train[va_ids] = full_pred
                q_nui_train[va_ids] = nui_pred
        default = float(labels.mean())
        q_full_train[~np.isfinite(q_full_train)] = default
        q_nui_train[~np.isfinite(q_nui_train)] = default
        q_full_valid, full_diag, full_model = _predict_timestamp_classifier(
            z,
            labels.to_numpy(dtype=np.int8),
            z_valid.reindex(columns=z.columns),
            seed=int(seed + 1907),
            max_depth=int(args.period_max_depth),
            n_estimators=int(args.period_n_estimators),
            min_child_fraction=float(args.period_min_child_fraction),
        )
        q_nui_valid, nui_diag, nui_model = _predict_timestamp_classifier(
            nui,
            labels.to_numpy(dtype=np.int8),
            nuisance_valid.reindex(columns=nui.columns),
            seed=int(seed + 2909),
            max_depth=2,
            n_estimators=max(80, int(args.period_n_estimators // 2)),
            min_child_fraction=0.05,
        )
        del full_model, nui_model
        train_inc_values = (_safe_logit(q_full_train) - _safe_logit(q_nui_train)).astype(np.float32, copy=False)
        valid_inc_values = (_safe_logit(q_full_valid) - _safe_logit(q_nui_valid)).astype(np.float32, copy=False)
        train_inc_ts = pd.Series(train_inc_values, index=labels.index, dtype="float32").reindex(z_train.index).fillna(0.0)
        valid_inc_ts = pd.Series(valid_inc_values, index=z_valid.index, dtype="float32")
        diag.update(
            {
                "period_reason": "",
                "period_full_feature_count": int(full_diag.get("feature_count", 0)),
                "period_nuisance_feature_count": int(nui_diag.get("feature_count", 0)),
                "period_full_inner_auc": stack._safe_auc(labels.to_numpy(dtype=np.int8), q_full_train),
                "period_nuisance_inner_auc": stack._safe_auc(labels.to_numpy(dtype=np.int8), q_nui_train),
            }
        )
        if np.isfinite(diag["period_full_inner_auc"]) and np.isfinite(diag["period_nuisance_inner_auc"]):
            diag["period_increment_inner_auc_delta"] = float(diag["period_full_inner_auc"] - diag["period_nuisance_inner_auc"])
    train_ts, valid_ts = _time_series_increment_features(train_inc_ts, valid_inc_ts, stem="q_period_inc")
    return stack._align_timestamp_features(train_timestamps, train_ts), stack._align_timestamp_features(valid_timestamps, valid_ts), diag


def _time_series_increment_features(train_score: pd.Series, valid_score: pd.Series, *, stem: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_score = train_score.sort_index().astype(float)
    valid_score = valid_score.sort_index().astype(float)
    combined = pd.concat([train_score, valid_score], axis=0).sort_index()
    out = pd.DataFrame(
        {
            stem: combined,
            f"{stem}_lag_12h": combined.shift(12),
            f"{stem}_lag_24h": combined.shift(24),
            f"{stem}_change_12h": combined - combined.shift(12),
            f"{stem}_recent_max_24h": combined.rolling(24, min_periods=1).max(),
        },
        index=combined.index,
    )
    return _downcast_numeric(out.reindex(train_score.index)), _downcast_numeric(out.reindex(valid_score.index))


def _limited_interactions(features: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=features.index)

    def add(name: str, left: str, right: str) -> None:
        if left in features.columns and right in features.columns:
            l = pd.to_numeric(features[left], errors="coerce").to_numpy(dtype=np.float32)
            r = pd.to_numeric(features[right], errors="coerce").to_numpy(dtype=np.float32)
            out[name] = (l * r).astype(np.float32, copy=False)

    add("q_period_x_q_fail", "q_period_inc", "q_fail_inc")
    add("q_period_x_prediction_support_quality", "q_period_inc", "prediction_support_quality")
    add("q_period_x_prediction_path_instability", "q_period_inc", "prediction_path_instability")
    add("q_fail_x_base_leaf_occupancy_novelty", "q_fail_inc", "base_leaf_leaf_occupancy_novelty")
    add("q_fail_x_meta_leaf_occupancy_novelty", "q_fail_inc", "anchor_meta_leaf_leaf_occupancy_novelty")
    add("q_fail_x_prediction_support_quality", "q_fail_inc", "prediction_support_quality")
    add("q_period_x_leverage_funding_crowding", "q_period_inc", "leverage_funding_crowding")
    add("q_period_x_liquidity_participation_stress", "q_period_inc", "liquidity_participation_stress")
    return _downcast_numeric(out)


def _fit_correction_model(
    *,
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: np.ndarray,
    anchor_train: np.ndarray,
    anchor_valid: np.ndarray,
    timestamps_train: pd.Series,
    seed: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if lgb is None:
        raise RuntimeError("lightgbm is required")
    x = pd.concat([x_train, x_valid], axis=0, ignore_index=True, copy=False).replace([np.inf, -np.inf], np.nan)
    keep_cols = [c for c in x_train.columns if pd.to_numeric(x_train[c], errors="coerce").notna().mean() > 0.02]
    if not keep_cols:
        return np.asarray(anchor_valid, dtype=np.float32), {"reason": "empty_correction_matrix", "feature_count": 0}
    prepared = _prepare_model_matrix(x.loc[:, keep_cols])
    x_tr = prepared.iloc[: len(x_train)]
    x_va = prepared.iloc[len(x_train) :]
    y_train = np.asarray(y_train, dtype=np.int8)
    valid_train = np.flatnonzero(y_train >= 0)
    if len(valid_train) < 200 or len(np.unique(y_train[valid_train])) < 2:
        return np.asarray(anchor_valid, dtype=np.float32), {"reason": "insufficient_rows_or_classes", "feature_count": int(len(keep_cols))}
    if int(args.max_train_rows) > 0 and len(valid_train) > int(args.max_train_rows):
        fit_ids = canon._period_stratified_train_sample(
            timestamps=timestamps_train.reset_index(drop=True),
            y=np.maximum(y_train, 0),
            train_idx=valid_train,
            max_rows=int(args.max_train_rows),
            seed=int(seed),
        )
    else:
        fit_ids = valid_train
    p0 = np.clip(np.asarray(anchor_train, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    denom = np.clip(p0 * (1.0 - p0), 1e-3, None)
    residual = np.clip((y_train.astype(np.float64) - p0) / denom, -2.0, 2.0).astype(np.float32)
    weight = denom.astype(np.float32)
    min_child = max(50, int(math.ceil(float(args.correction_min_child_fraction) * len(fit_ids))))
    constraints = [-1 if "q_fail" in str(col) else 0 for col in keep_cols]
    reg = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=int(args.correction_n_estimators),
        learning_rate=float(args.correction_learning_rate),
        max_depth=int(args.correction_max_depth),
        num_leaves=max(4, min(8, 2 ** int(args.correction_max_depth))),
        min_child_samples=int(min_child),
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=5.0,
        monotone_constraints=constraints,
        random_state=int(seed),
        n_jobs=max(1, min(6, os.cpu_count() or 2)),
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(
            x_tr.iloc[fit_ids],
            residual[fit_ids],
            sample_weight=weight[fit_ids],
        )
    delta = _clip_delta(reg.predict(x_va).astype(np.float32, copy=False), float(args.correction_clip))
    pred = _sigmoid(_safe_logit(anchor_valid) + delta)
    return pred, {
        "reason": "",
        "feature_count": int(len(keep_cols)),
        "train_rows": int(len(fit_ids)),
        "min_child_samples": int(min_child),
        "delta_mean": float(np.nanmean(delta)),
        "delta_std": float(np.nanstd(delta)),
        "delta_clip": float(args.correction_clip),
        "monotone_q_fail_constraints": int(sum(1 for c in constraints if c < 0)),
    }


def _direct_scores(anchor_valid: np.ndarray, features_valid: pd.DataFrame, args: argparse.Namespace) -> dict[str, np.ndarray]:
    q_fail = pd.to_numeric(features_valid.get("q_fail_inc"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    q_period = pd.to_numeric(features_valid.get("q_period_inc"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    r1_delta = -float(args.r1_lambda_fail) * q_fail
    r2_delta = -float(args.r2_lambda_fail) * q_fail - float(args.r2_gamma_period) * q_period
    r3_delta = -(
        float(args.r3_lambda0_fail) + float(args.r3_lambda1_period_fail) * np.clip(q_period, -2.0, 2.0)
    ) * q_fail - float(args.r3_gamma_period) * q_period
    return {
        TRIAL_R1: _score_with_delta(anchor_valid, r1_delta, float(args.correction_clip)),
        TRIAL_R2: _score_with_delta(anchor_valid, r2_delta, float(args.correction_clip)),
        TRIAL_R3: _score_with_delta(anchor_valid, r3_delta, float(args.correction_clip)),
    }


def _write_report(out_dir: Path, summary: pd.DataFrame, directional: pd.DataFrame, fold_diag: pd.DataFrame) -> None:
    lines = [
        "# Anchored Reliability Meta Correction",
        "",
        "One unchanged `y_bin` meta output is produced by correcting the frozen anchor score in logit space.",
        "",
        "## Validation Contract",
        "",
        "- `meta_0` is the current OOF meta score.",
        "- Failure score is incremental: `logit(q_full) - logit(q_controls)` inside the rank>=50% anchor population.",
        "- Difficult-period score is incremental versus nuisance coverage/session/universe predictors.",
        "- Leaf/context inputs are fold-fitted and do not use realized leaf outcomes.",
        "- `meta_1` corrections are clipped and monotone-decreasing in `q_fail*` features.",
        "",
        "## Summary",
        "",
    ]
    if summary.empty:
        lines.append("_No summary rows._")
    else:
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
            "top10_delta_mean_return",
            "top10_delta_lower_tail_return",
        ]
        lines.append(summary[[c for c in cols if c in summary.columns]].to_markdown(index=False, floatfmt=".5f"))
    if not directional.empty:
        lines.extend(["", "## Directional Metrics", ""])
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
            "worst_week_hr_top30",
            "q10_week_hr_top30",
        ]
        lines.append(directional[[c for c in cols if c in directional.columns]].to_markdown(index=False, floatfmt=".5f"))
    if not fold_diag.empty:
        lines.extend(["", "## Fold Diagnostics Snapshot", ""])
        lines.append(fold_diag.head(60).to_markdown(index=False, floatfmt=".5f"))
    (out_dir / "anchored_reliability_meta_correction_report.md").write_text("\n".join(lines))


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
    directional_timestamp_frames: list[pd.DataFrame] = []
    directional_summary_rows: list[dict[str, Any]] = []
    directional_episode_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []

    for head in heads:
        print(f"[anchored_reliability] head={head.head}", flush=True)
        panel = _downcast_numeric(_normalise_keys(pd.read_parquet(head.meta_oof_path)), exclude=["timestamp", "symbol"])
        panel = panel.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
        if int(args.max_rows) > 0 and len(panel) > int(args.max_rows):
            keep = np.linspace(0, len(panel) - 1, int(args.max_rows)).round().astype(int)
            panel = panel.iloc[np.unique(keep)].sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
            print(f"[anchored_reliability] sampled head={head.head} rows={len(panel)}", flush=True)
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
        preds: dict[str, np.ndarray] = {trial: np.full(len(panel), np.nan, dtype=np.float32) for trial in TRIALS}
        q_feature_frames: list[pd.DataFrame] = []

        path_cols: list[str] | None = None
        for fold in folds:
            tr = np.asarray(fold.train_idx, dtype=np.int64)
            va = np.asarray(fold.valid_idx, dtype=np.int64)
            fold_valid_mask[va] = True
            ts_train = panel["timestamp"].iloc[tr].reset_index(drop=True)
            ts_valid = panel["timestamp"].iloc[va].reset_index(drop=True)
            raw_train = raw.iloc[tr].reset_index(drop=True)
            raw_valid = raw.iloc[va].reset_index(drop=True)
            controls_train, path_cols = _anchor_control_features(
                raw_train,
                ts_train,
                anchor[tr],
                max_path_cols=int(args.max_control_path_features),
                selected_cols=path_cols,
            )
            controls_valid, _ = _anchor_control_features(
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
            full_train = stack._combine_features(controls_train, canonical_train, leaf_train)
            full_valid = stack._combine_features(controls_valid, canonical_valid, leaf_valid)
            q_fail_train, q_fail_valid, fail_diag = _crossfit_failure_increment(
                controls_train=controls_train,
                full_train=full_train,
                controls_valid=controls_valid,
                full_valid=full_valid,
                train_timestamps=ts_train,
                y_train=y[tr],
                anchor_train=anchor[tr],
                seed=int(args.seed + 101 * fold.fold_id),
                args=args,
            )
            q_fail_train_features = _rank_features_from_values(ts_train, q_fail_train, "q_fail_inc")
            q_fail_valid_features = _rank_features_from_values(ts_valid, q_fail_valid, "q_fail_inc")
            z_source_train = stack._combine_features(canonical_train, current_x.iloc[tr].reset_index(drop=True))
            z_source_valid = stack._combine_features(canonical_valid, current_x.iloc[va].reset_index(drop=True))
            z_train = stack._timestamp_feature_table(z_source_train, ts_train, max_columns=int(args.max_timestamp_features))
            z_valid = stack._timestamp_feature_table(z_source_valid, ts_valid, max_columns=int(args.max_timestamp_features))
            z_valid = z_valid.reindex(columns=z_train.columns)
            nui_train = _timestamp_nuisance_table(ts_train, panel["symbol"].iloc[tr] if "symbol" in panel.columns else None)
            nui_valid = _timestamp_nuisance_table(ts_valid, panel["symbol"].iloc[va] if "symbol" in panel.columns else None)
            period_train, period_valid, period_diag = _period_increment_features(
                z_train=z_train,
                z_valid=z_valid,
                nuisance_train=nui_train,
                nuisance_valid=nui_valid,
                train_timestamps=ts_train,
                valid_timestamps=ts_valid,
                y_train=y[tr],
                anchor_train=anchor[tr],
                seed=int(args.seed + 211 * fold.fold_id),
                args=args,
            )
            anchor_train_features = controls_train.loc[:, ["anchor_p0", "anchor_logit0", "anchor_rank0_by_timestamp"]]
            anchor_valid_features = controls_valid.loc[:, ["anchor_p0", "anchor_logit0", "anchor_rank0_by_timestamp"]]
            main_train = stack._combine_features(anchor_train_features, q_fail_train_features, period_train, canonical_train, leaf_train)
            main_valid = stack._combine_features(anchor_valid_features, q_fail_valid_features, period_valid, canonical_valid, leaf_valid)
            inter_train = stack._combine_features(main_train, _limited_interactions(main_train))
            inter_valid = stack._combine_features(main_valid, _limited_interactions(main_valid))
            direct = _direct_scores(anchor[va], main_valid, args)
            preds[TRIAL_ANCHOR][va] = anchor[va]
            for trial, values in direct.items():
                preds[trial][va] = values
            main_pred, main_diag = _fit_correction_model(
                x_train=main_train,
                x_valid=main_valid,
                y_train=y[tr],
                anchor_train=anchor[tr],
                anchor_valid=anchor[va],
                timestamps_train=ts_train,
                seed=int(args.seed + 307 * fold.fold_id),
                args=args,
            )
            inter_pred, inter_diag = _fit_correction_model(
                x_train=inter_train,
                x_valid=inter_valid,
                y_train=y[tr],
                anchor_train=anchor[tr],
                anchor_valid=anchor[va],
                timestamps_train=ts_train,
                seed=int(args.seed + 409 * fold.fold_id),
                args=args,
            )
            preds[TRIAL_META_MAIN][va] = main_pred
            preds[TRIAL_META_INTERACTIONS][va] = inter_pred
            q_fold = pd.DataFrame(
                {
                    "row_id": va,
                    "q_fail_inc": pd.to_numeric(main_valid.get("q_fail_inc"), errors="coerce").to_numpy(dtype=np.float32),
                    "q_period_inc": pd.to_numeric(main_valid.get("q_period_inc"), errors="coerce").to_numpy(dtype=np.float32),
                }
            )
            q_feature_frames.append(q_fold)
            common = {
                "head": head.head,
                "fold": int(fold.fold_id),
                "train_rows": int(len(tr)),
                "valid_rows": int(len(va)),
                "path_control_feature_count": int(len(path_cols or [])),
                **canonical_diag,
                **fail_diag,
                **meta_leaf_diag,
                **base_leaf_diag,
                **period_diag,
            }
            fold_rows.append({**common, "trial": TRIAL_META_MAIN, **{f"correction_{k}": v for k, v in main_diag.items()}})
            fold_rows.append({**common, "trial": TRIAL_META_INTERACTIONS, **{f"correction_{k}": v for k, v in inter_diag.items()}})
            print(
                f"[anchored_reliability] head={head.head} fold={fold.fold_id}/{len(folds)} "
                f"train={len(tr)} valid={len(va)}",
                flush=True,
            )

        baseline_fold = anchor.copy()
        baseline_fold[~fold_valid_mask] = np.nan
        for trial, pred in preds.items():
            summary = ctx._overall_metrics(
                head=head.head,
                arm=trial,
                variant="anchored_reliability_correction" if trial != TRIAL_ANCHOR else "anchor_reference",
                y=y,
                pred=pred,
                baseline_pred=baseline_fold,
                returns=returns,
            )
            summary.update(
                {
                    "training_target": "y_bin",
                    "single_output_score": True,
                    "anchor_trial": TRIAL_ANCHOR,
                    "correction_clipped": trial in {TRIAL_R1, TRIAL_R2, TRIAL_R3, TRIAL_META_MAIN, TRIAL_META_INTERACTIONS},
                    "correction_clip": float(args.correction_clip) if trial != TRIAL_ANCHOR else 0.0,
                    "forbidden_targets_used": False,
                }
            )
            summary_rows.append(summary)
            ts_metrics = ctx._directional_timestamp_metrics(
                head=head.head,
                arm=trial,
                variant=str(summary["distillation_variant"]),
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
                    directional_summary_rows.extend(agg.to_dict(orient="records"))
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
        for trial, pred in preds.items():
            score_frame[trial] = pred
        if q_feature_frames:
            q_all = pd.concat(q_feature_frames, axis=0, ignore_index=True).drop_duplicates("row_id")
            score_frame = score_frame.merge(q_all, on="row_id", how="left")
        score_frames.append(score_frame)

    summary = pd.DataFrame(summary_rows)
    directional = (
        pd.DataFrame(directional_summary_rows)
        if directional_summary_rows
        else pd.DataFrame(columns=DIRECTIONAL_SUMMARY_COLUMNS)
    )
    directional_timestamp = (
        pd.concat(directional_timestamp_frames, axis=0, ignore_index=True)
        if directional_timestamp_frames
        else pd.DataFrame(columns=DIRECTIONAL_TIMESTAMP_COLUMNS)
    )
    directional_episode = (
        pd.concat(directional_episode_frames, axis=0, ignore_index=True)
        if directional_episode_frames
        else pd.DataFrame(
            columns=[
                "head",
                "arm",
                "distillation_variant",
                "heldout_episode",
                "period_type",
                "delta_timestamp_weighted_hr_top30",
                "delta_ndcg_top30",
            ]
        )
    )
    directional_episode_ci = ctx._directional_episode_block_confidence_intervals(
        directional_episode,
        seed=int(args.seed),
        bootstrap_rounds=int(args.bootstrap_rounds),
    )
    if directional_episode_ci.empty:
        directional_episode_ci = pd.DataFrame(
            columns=[
                "head",
                "arm",
                "distillation_variant",
                "metric",
                "episode_count",
                "mean",
                "median",
                "positive_episode_rate",
                "ci05",
                "ci95",
                "ci_method",
            ]
        )
    fold_diag = pd.DataFrame(fold_rows)
    scores = pd.concat(score_frames, axis=0, ignore_index=True) if score_frames else pd.DataFrame()

    summary.to_csv(out_dir / "anchored_reliability_meta_correction_summary.csv", index=False)
    directional.to_csv(out_dir / "anchored_reliability_directional_metrics.csv", index=False)
    directional_timestamp.to_csv(out_dir / "anchored_reliability_directional_timestamp_metrics.csv", index=False)
    directional_episode.to_csv(out_dir / "anchored_reliability_directional_episode_metrics.csv", index=False)
    directional_episode_ci.to_csv(out_dir / "anchored_reliability_directional_episode_ci.csv", index=False)
    fold_diag.to_csv(out_dir / "anchored_reliability_fold_diagnostics.csv", index=False)
    if not scores.empty:
        scores.to_parquet(out_dir / "anchored_reliability_oof_scores.parquet", index=False)
    manifest = {
        "output_dir": str(out_dir),
        "meta_artifact_dir": str(meta_artifact_dir),
        "baseline_artifact_dir": str(baseline_artifact_dir),
        "feature_dir": str(feature_dir),
        "heads": [h.head for h in heads],
        "trials": list(TRIALS),
        "contract": {
            "target": "y_bin",
            "single_output_score": True,
            "anchor": "current_meta_oof_score",
            "correction_form": "z_final = z0 + clipped_delta",
            "failure_signal": "incremental_logit_full_minus_controls_top50_anchor_population",
            "period_signal": "incremental_logit_full_minus_nuisance_timestamp_classifier",
        },
        "args": vars(args),
    }
    (out_dir / "anchored_reliability_manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default))
    _write_report(out_dir, summary, directional, fold_diag)
    print(f"[anchored_reliability] wrote {out_dir}", flush=True)
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
    parser.add_argument("--output-dir", default="data_perp/reports/anchored_reliability_meta_correction_20260623")
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
    parser.add_argument("--max-timestamp-features", type=int, default=80)
    parser.add_argument("--failure-candidate-rank-threshold", type=float, default=0.50)
    parser.add_argument("--failure-min-train-rows", type=int, default=200)
    parser.add_argument("--failure-max-depth", type=int, default=3)
    parser.add_argument("--failure-n-estimators", type=int, default=180)
    parser.add_argument("--failure-min-child-fraction", type=float, default=0.025)
    parser.add_argument("--period-short-window", type=int, default=72)
    parser.add_argument("--period-long-window", type=int, default=120)
    parser.add_argument("--period-difficult-quantile", type=float, default=0.25)
    parser.add_argument("--period-min-train-timestamps", type=int, default=80)
    parser.add_argument("--period-max-depth", type=int, default=3)
    parser.add_argument("--period-n-estimators", type=int, default=160)
    parser.add_argument("--period-min-child-fraction", type=float, default=0.035)
    parser.add_argument("--leaf-max-models", type=int, default=1)
    parser.add_argument("--leaf-tree-stride", type=int, default=3)
    parser.add_argument("--leaf-max-trees", type=int, default=120)
    parser.add_argument("--r1-lambda-fail", type=float, default=0.25)
    parser.add_argument("--r2-lambda-fail", type=float, default=0.25)
    parser.add_argument("--r2-gamma-period", type=float, default=0.10)
    parser.add_argument("--r3-lambda0-fail", type=float, default=0.20)
    parser.add_argument("--r3-lambda1-period-fail", type=float, default=0.10)
    parser.add_argument("--r3-gamma-period", type=float, default=0.10)
    parser.add_argument("--correction-clip", type=float, default=0.50)
    parser.add_argument("--correction-max-depth", type=int, default=2)
    parser.add_argument("--correction-n-estimators", type=int, default=140)
    parser.add_argument("--correction-learning-rate", type=float, default=0.035)
    parser.add_argument("--correction-min-child-fraction", type=float, default=0.025)
    parser.add_argument("--rank-threshold", type=float, default=0.70)
    parser.add_argument("--directional-min-timestamp-rows", type=int, default=5)
    parser.add_argument("--bootstrap-rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
