#!/usr/bin/env python3
"""Run one-head contextual meta ablations with the unchanged meta label.

This experiment replaces the earlier failure-head retraining framing.  It keeps
one meta model, one output score, and the current `y_bin` target for each
strategy head.  Failure classifiers, leaf interactions, and archetype
diagnostics are used only to choose ordinary input features.
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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from scripts.diagnose_meta_recent_failures import (
    _base_models_for_head,
    _candidate_feature_contract,
    _discover_heads,
    _downcast_numeric,
    _feature_store_union,
    _known_export_features,
    _merge_feature_candidates,
    _normalise_keys,
    _prepare_model_matrix,
    _read_regime_features,
    _weekly_high_conf_metrics,
    _bad_recent_weeks,
    lgb,
)
from scripts.quantify_bad_regime_archetype_usefulness import _pick_realized_return
from scripts.run_canonical_context_retrain_experiment import (
    FoldContext,
    MARKET_STATE,
    MODEL_STATE as BASE_MODEL_STATE,
    _assemble_selected_matrix,
    _calibration_slope_intercept,
    _fold_canonical_features,
    _fresh_oos_canonical_features,
    _fresh_oos_indices,
    _load_canonical_definitions,
    _make_chrono_folds,
    _period_stratified_train_sample,
    _safe_auc,
    _safe_pr_auc,
)


DEFAULT_EPISODE_REGISTRY = Path("data_perp/reports/contextual_meta_episode_registry_20260622/frozen_bad_episode_registry.csv")
FORBIDDEN_TRAINING_TARGETS = {
    "high_conf_miss",
    "high_conf_tail_loss",
    "failure",
    "tail_loss",
    "expected_return",
    "payoff_quantile",
}
HEADS = ("long_dist", "short_boll", "short_asset", "long_bars")
MODEL_STATE = BASE_MODEL_STATE + ("leaf_occupancy_novelty",)
CANONICAL_CONTEXT = MODEL_STATE + MARKET_STATE
ARM_A = "A_current_meta_model"
ARM_B = "B_current_plus_model_state"
ARM_C = "C_current_plus_market_state"
ARM_D = "D_current_plus_both_contexts"
ARM_E = "E_current_plus_targeted_interactions"
ARM_G = "G_rank_preserving_contextual_calibration"
ARM_H = "H_regularized_contextual_correction"
ARM_I = "I_regime_balanced_targeted_interactions"
ARM_J = "J_oracle_routed_bad_regime_specialist_diagnostic"
FEATURE_ARMS = (ARM_A, ARM_B, ARM_C, ARM_D, ARM_E)
CONTEXTUAL_SCORE_ARMS = (ARM_G, ARM_H)
DIAGNOSTIC_ARMS = (ARM_I,)
DISTILLATION_VARIANTS = (
    "current_self_distillation",
    "hard_label_only",
    "lower_global_distillation_weight",
    "support_aware_distillation",
    "context_aware_distillation",
    "support_x_context_aware_distillation",
)
INTERACTION_PAIRS = (
    ("prediction_support_quality", "leverage_funding_crowding"),
    ("prediction_support_quality", "liquidity_participation_stress"),
    ("prediction_path_instability", "tail_volatility_stress"),
    ("prediction_reconstruction_anomaly", "relative_value_dislocation"),
    ("regime_similarity_or_novelty", "leverage_funding_crowding"),
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _meta_target(panel: pd.DataFrame) -> np.ndarray:
    if "y_bin" not in panel.columns:
        raise RuntimeError("Current unchanged meta label `y_bin` is missing")
    y = pd.to_numeric(panel["y_bin"], errors="coerce").to_numpy(dtype=np.float32)
    finite = np.isfinite(y)
    y_bin = np.full(len(y), -1, dtype=np.int8)
    y_bin[finite] = (y[finite] > 0.5).astype(np.int8)
    return y_bin


def _current_meta_score(panel: pd.DataFrame) -> np.ndarray:
    for col in ("oof_pred", "oof_meta_clf", "oof_p_move"):
        if col in panel.columns:
            score = pd.to_numeric(panel[col], errors="coerce").to_numpy(dtype=np.float32)
            return np.clip(score, 1e-6, 1.0 - 1e-6).astype(np.float32, copy=False)
    raise RuntimeError("No current meta score column found")


def _safe_log_loss(y: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(pred) & (y >= 0)
    if int(mask.sum()) < 30 or len(np.unique(y[mask])) < 2:
        return np.nan
    return float(log_loss(y[mask], np.clip(pred[mask], 1e-6, 1.0 - 1e-6), labels=[0, 1]))


def _safe_brier(y: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(pred) & (y >= 0)
    if int(mask.sum()) < 30 or len(np.unique(y[mask])) < 2:
        return np.nan
    return float(brier_score_loss(y[mask], np.clip(pred[mask], 1e-6, 1.0 - 1e-6)))


def _safe_pr(y: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(pred) & (y >= 0)
    if int(mask.sum()) < 30 or len(np.unique(y[mask])) < 2:
        return np.nan
    return float(average_precision_score(y[mask], pred[mask]))


def _return_metrics(returns: np.ndarray) -> dict[str, Any]:
    ret = np.asarray(returns, dtype=np.float64)
    ret = ret[np.isfinite(ret)]
    if ret.size == 0:
        return {
            "trade_count": 0,
            "mean_return": np.nan,
            "winner_magnitude": np.nan,
            "loser_magnitude": np.nan,
            "lower_tail_return": np.nan,
        }
    winners = ret[ret > 0.0]
    losers = ret[ret < 0.0]
    return {
        "trade_count": int(ret.size),
        "mean_return": float(np.nanmean(ret)),
        "winner_magnitude": float(np.nanmean(winners)) if winners.size else 0.0,
        "loser_magnitude": float(np.nanmean(np.abs(losers))) if losers.size else 0.0,
        "lower_tail_return": float(np.nanquantile(ret, 0.05)) if ret.size >= 20 else np.nan,
    }


def _overall_metrics(
    *,
    head: str,
    arm: str,
    variant: str,
    y: np.ndarray,
    pred: np.ndarray,
    baseline_pred: np.ndarray,
    returns: np.ndarray,
) -> dict[str, Any]:
    mask = (y >= 0) & np.isfinite(pred) & np.isfinite(baseline_pred)
    yy = y[mask]
    pp = np.clip(pred[mask], 1e-6, 1.0 - 1e-6)
    bb = np.clip(baseline_pred[mask], 1e-6, 1.0 - 1e-6)
    rr = returns[mask]
    slope, intercept = _calibration_slope_intercept(yy.astype(np.float32), pp.astype(np.float32))
    base_slope, base_intercept = _calibration_slope_intercept(yy.astype(np.float32), bb.astype(np.float32))
    row = {
        "head": head,
        "arm": arm,
        "distillation_variant": variant,
        "rows": int(mask.sum()),
        "label_positive_rate": float(np.mean(yy)) if yy.size else np.nan,
        "auc": _safe_auc(yy.astype(np.float32), pp.astype(np.float32)),
        "baseline_auc": _safe_auc(yy.astype(np.float32), bb.astype(np.float32)),
        "pr_auc": _safe_pr(yy, pp),
        "baseline_pr_auc": _safe_pr(yy, bb),
        "log_loss": _safe_log_loss(yy, pp),
        "baseline_log_loss": _safe_log_loss(yy, bb),
        "brier": _safe_brier(yy, pp),
        "baseline_brier": _safe_brier(yy, bb),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "baseline_calibration_slope": base_slope,
        "baseline_calibration_intercept": base_intercept,
        "scored_coverage": float(np.mean(mask)) if len(mask) else 0.0,
    }
    row["delta_log_loss_improvement"] = row["baseline_log_loss"] - row["log_loss"]
    row["delta_brier_improvement"] = row["baseline_brier"] - row["brier"]
    row["delta_auc"] = row["auc"] - row["baseline_auc"]
    row["delta_pr_auc"] = row["pr_auc"] - row["baseline_pr_auc"]
    row.update({f"all_{k}": v for k, v in _return_metrics(rr).items()})
    for coverage in (0.10, 0.20, 0.30):
        n = max(1, int(math.ceil(float(coverage) * len(pp))))
        pick = np.argsort(pp)[::-1][:n]
        base_pick = np.argsort(bb)[::-1][:n]
        cur_ret = _return_metrics(rr[pick])
        base_ret = _return_metrics(rr[base_pick])
        prefix = f"top{int(coverage * 100)}"
        for key, value in cur_ret.items():
            row[f"{prefix}_{key}"] = value
        row[f"{prefix}_hit_rate"] = float(np.mean(yy[pick])) if len(pick) else np.nan
        row[f"{prefix}_baseline_mean_return"] = base_ret["mean_return"]
        row[f"{prefix}_delta_mean_return"] = cur_ret["mean_return"] - base_ret["mean_return"]
        row[f"{prefix}_delta_winner_magnitude"] = cur_ret["winner_magnitude"] - base_ret["winner_magnitude"]
        row[f"{prefix}_delta_lower_tail_return"] = cur_ret["lower_tail_return"] - base_ret["lower_tail_return"]
    return row


def _eligible_directional_mask(
    panel: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
    baseline_pred: np.ndarray,
    *,
    rank_threshold: float,
) -> tuple[np.ndarray, str]:
    ts = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
    mask = (y >= 0) & np.isfinite(pred) & np.isfinite(baseline_pred) & ts.notna().to_numpy(dtype=bool)
    source = "valid_meta_rows"
    if "oof_rank_pct" in panel.columns:
        rank = pd.to_numeric(panel["oof_rank_pct"], errors="coerce").to_numpy(dtype=np.float32)
        high = np.isfinite(rank) & (rank >= float(rank_threshold))
        if int((mask & high).sum()) > 0:
            mask &= high
            source = f"oof_rank_pct>={float(rank_threshold):.4f}"
    return mask, source


def _ranking_stats(labels: np.ndarray, scores: np.ndarray, coverage: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    valid = np.isfinite(scores) & np.isfinite(labels)
    ids = np.flatnonzero(valid)
    if ids.size == 0:
        return {
            "local_idx": np.array([], dtype=np.int64),
            "count": 0,
            "hits": 0.0,
            "hit_rate": np.nan,
            "dcg": np.nan,
            "ndcg": np.nan,
            "average_precision": np.nan,
            "pairwise_concordance": np.nan,
        }
    order = ids[np.argsort(scores[ids], kind="mergesort")[::-1]]
    k = max(1, int(math.ceil(float(coverage) * len(order))))
    top = order[:k]
    top_y = labels[top].astype(np.float64, copy=False)
    denom = np.log2(np.arange(2, len(top_y) + 2, dtype=np.float64))
    dcg = float(np.sum(top_y / denom)) if len(top_y) else np.nan
    ideal_y = np.sort(labels[ids].astype(np.float64, copy=False))[::-1][:k]
    ideal_dcg = float(np.sum(ideal_y / denom)) if len(ideal_y) else 0.0
    positives_possible = int(np.sum(labels[ids] > 0.5))
    if positives_possible > 0 and ideal_dcg > 0.0:
        ndcg = dcg / ideal_dcg
    else:
        ndcg = np.nan
    positives_at_k = top_y > 0.5
    ap_denom = min(positives_possible, k)
    if ap_denom > 0:
        precision_at_i = np.cumsum(positives_at_k) / np.arange(1, len(top_y) + 1, dtype=np.float64)
        average_precision = float(np.sum(precision_at_i[positives_at_k]) / float(ap_denom))
    else:
        average_precision = np.nan
    pos = int(np.sum(positives_at_k))
    neg = int(len(top_y) - pos)
    if pos > 0 and neg > 0:
        positives_seen = 0
        concordant = 0
        for value in positives_at_k:
            if value:
                positives_seen += 1
            else:
                concordant += positives_seen
        pairwise = float(concordant / float(pos * neg))
    else:
        pairwise = np.nan
    return {
        "local_idx": top.astype(np.int64, copy=False),
        "count": int(len(top)),
        "hits": float(np.sum(top_y)),
        "hit_rate": float(np.mean(top_y)) if len(top_y) else np.nan,
        "dcg": dcg,
        "ndcg": float(ndcg) if np.isfinite(ndcg) else np.nan,
        "average_precision": float(average_precision) if np.isfinite(average_precision) else np.nan,
        "pairwise_concordance": float(pairwise) if np.isfinite(pairwise) else np.nan,
    }


def _log_loss_sum(y: np.ndarray, pred: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.float64)
    pp = np.clip(np.asarray(pred, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return float(-np.sum(yy * np.log(pp) + (1.0 - yy) * np.log(1.0 - pp)))


def _brier_sum(y: np.ndarray, pred: np.ndarray) -> float:
    yy = np.asarray(y, dtype=np.float64)
    pp = np.clip(np.asarray(pred, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return float(np.sum((pp - yy) ** 2))


def _band_hit_rate(labels: np.ndarray, scores: np.ndarray, start_frac: float, end_frac: float) -> tuple[float, int, float]:
    labels = np.asarray(labels, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    valid = np.isfinite(scores) & np.isfinite(labels)
    ids = np.flatnonzero(valid)
    if ids.size == 0:
        return np.nan, 0, 0.0
    order = ids[np.argsort(scores[ids], kind="mergesort")[::-1]]
    start = int(math.ceil(float(start_frac) * len(order)))
    end = int(math.ceil(float(end_frac) * len(order)))
    start = min(max(start, 0), len(order))
    end = min(max(end, start), len(order))
    if end <= start:
        return np.nan, 0, 0.0
    picked = order[start:end]
    yy = labels[picked].astype(np.float64, copy=False)
    return float(np.mean(yy)), int(len(picked)), float(np.sum(yy))


def _directional_timestamp_metrics(
    *,
    head: str,
    arm: str,
    variant: str,
    panel: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
    baseline_pred: np.ndarray,
    returns: np.ndarray,
    bad_episodes: set[str],
    rank_threshold: float,
    min_timestamp_rows: int,
) -> pd.DataFrame:
    pred_arr = np.asarray(pred, dtype=np.float32)
    base_arr = np.asarray(baseline_pred, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int8)
    returns_arr = np.asarray(returns, dtype=np.float32)
    eligible, eligibility_source = _eligible_directional_mask(
        panel,
        y_arr,
        pred_arr,
        base_arr,
        rank_threshold=float(rank_threshold),
    )
    ts = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce").reset_index(drop=True)
    episodes = _episode_labels(panel).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    idx_all = np.flatnonzero(eligible)
    if idx_all.size == 0:
        return pd.DataFrame()
    frame = pd.DataFrame({"idx": idx_all, "timestamp": ts.iloc[idx_all].to_numpy()})
    for timestamp, group in frame.groupby("timestamp", sort=True):
        ids = group["idx"].to_numpy(dtype=np.int64)
        if len(ids) < int(min_timestamp_rows):
            continue
        yy = y_arr[ids].astype(np.float32, copy=False)
        pp = pred_arr[ids]
        bb = base_arr[ids]
        rr = returns_arr[ids]
        row: dict[str, Any] = {
            "head": head,
            "arm": arm,
            "distillation_variant": variant,
            "timestamp": pd.Timestamp(timestamp).isoformat(),
            "week": pd.Timestamp(timestamp).to_period("W").start_time.strftime("%Y-%m-%d"),
            "eligible_rows": int(len(ids)),
            "eligibility_source": eligibility_source,
            "rank_threshold": float(rank_threshold),
            "min_timestamp_rows": int(min_timestamp_rows),
        }
        row["period_type"] = "bad_period" if str(episodes.iloc[ids[0]]) in set(str(x) for x in bad_episodes) else "normal_period"
        row["episode"] = str(episodes.iloc[ids[0]])
        top_indices: dict[str, np.ndarray] = {}
        base_top_indices: dict[str, np.ndarray] = {}
        for coverage in (0.10, 0.20, 0.30):
            pct = int(round(float(coverage) * 100))
            cur = _ranking_stats(yy, pp, coverage)
            base = _ranking_stats(yy, bb, coverage)
            cur_idx = cur["local_idx"]
            base_idx = base["local_idx"]
            top_indices[f"top{pct}"] = cur_idx
            base_top_indices[f"top{pct}"] = base_idx
            row[f"selected_count_top{pct}"] = int(cur["count"])
            row[f"hit_count_top{pct}"] = float(cur["hits"])
            row[f"hr_top{pct}"] = cur["hit_rate"]
            row[f"baseline_selected_count_top{pct}"] = int(base["count"])
            row[f"baseline_hit_count_top{pct}"] = float(base["hits"])
            row[f"baseline_hr_top{pct}"] = base["hit_rate"]
            row[f"delta_hr_top{pct}"] = (
                float(cur["hit_rate"] - base["hit_rate"])
                if np.isfinite(cur["hit_rate"]) and np.isfinite(base["hit_rate"])
                else np.nan
            )
            if pct == 30:
                for metric in ("dcg", "ndcg", "average_precision", "pairwise_concordance"):
                    row[f"{metric}_top30"] = cur[metric]
                    row[f"baseline_{metric}_top30"] = base[metric]
                    row[f"delta_{metric}_top30"] = (
                        float(cur[metric] - base[metric])
                        if np.isfinite(cur[metric]) and np.isfinite(base[metric])
                        else np.nan
                    )
        for label, start, end in (("90_100", 0.0, 0.10), ("80_90", 0.10, 0.20), ("70_80", 0.20, 0.30)):
            hr, count, hits = _band_hit_rate(yy, pp, start, end)
            base_hr, base_count, base_hits = _band_hit_rate(yy, bb, start, end)
            row[f"hr_{label}"] = hr
            row[f"selected_count_{label}"] = int(count)
            row[f"hit_count_{label}"] = float(hits)
            row[f"baseline_hr_{label}"] = base_hr
            row[f"baseline_selected_count_{label}"] = int(base_count)
            row[f"baseline_hit_count_{label}"] = float(base_hits)
            row[f"delta_hr_{label}"] = float(hr - base_hr) if np.isfinite(hr) and np.isfinite(base_hr) else np.nan
        cur_top30 = top_indices.get("top30", np.array([], dtype=np.int64))
        base_top30 = base_top_indices.get("top30", np.array([], dtype=np.int64))
        cur_set = set(cur_top30.tolist())
        base_set = set(base_top30.tolist())
        entrants = np.array(sorted(cur_set - base_set), dtype=np.int64)
        removals = np.array(sorted(base_set - cur_set), dtype=np.int64)
        union = cur_set | base_set
        row["top30_overlap_count"] = int(len(cur_set & base_set))
        row["top30_jaccard"] = float(len(cur_set & base_set) / len(union)) if union else np.nan
        row["top30_entrant_count"] = int(len(entrants))
        row["top30_removed_count"] = int(len(removals))
        row["top30_entrant_hit_rate"] = float(np.mean(yy[entrants])) if len(entrants) else np.nan
        row["top30_removed_hit_rate"] = float(np.mean(yy[removals])) if len(removals) else np.nan
        row["top30_entrant_hits"] = float(np.sum(yy[entrants])) if len(entrants) else 0.0
        row["top30_removed_hits"] = float(np.sum(yy[removals])) if len(removals) else 0.0
        row["net_correct_trades_gained"] = row["top30_entrant_hits"] - row["top30_removed_hits"]
        if len(cur_top30):
            cur_y = yy[cur_top30]
            cur_pred = pp[cur_top30]
            cur_base_pred = bb[cur_top30]
            cur_ret = rr[cur_top30]
            ret = _return_metrics(cur_ret)
            row["top30_log_loss_sum"] = _log_loss_sum(cur_y, cur_pred)
            row["top30_baseline_on_selected_log_loss_sum"] = _log_loss_sum(cur_y, cur_base_pred)
            row["top30_brier_sum"] = _brier_sum(cur_y, cur_pred)
            row["top30_baseline_on_selected_brier_sum"] = _brier_sum(cur_y, cur_base_pred)
            for key, value in ret.items():
                row[f"top30_{key}"] = value
        if len(base_top30):
            base_ret = _return_metrics(rr[base_top30])
            for key, value in base_ret.items():
                row[f"baseline_top30_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _weighted_rate(hits: pd.Series, count: pd.Series) -> float:
    denom = float(pd.to_numeric(count, errors="coerce").fillna(0.0).sum())
    if denom <= 0.0:
        return np.nan
    return float(pd.to_numeric(hits, errors="coerce").fillna(0.0).sum() / denom)


def _directional_aggregate(timestamp_metrics: pd.DataFrame) -> pd.DataFrame:
    if timestamp_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (head, arm, variant), group in timestamp_metrics.groupby(["head", "arm", "distillation_variant"], sort=True):
        row: dict[str, Any] = {
            "head": head,
            "arm": arm,
            "distillation_variant": variant,
            "directional_timestamp_count": int(len(group)),
            "directional_eligible_rows": int(pd.to_numeric(group["eligible_rows"], errors="coerce").fillna(0).sum()),
            "directional_eligibility_source": "|".join(sorted(set(group["eligibility_source"].astype(str)))),
        }
        for pct in (10, 20, 30):
            row[f"timestamp_weighted_hr_top{pct}"] = float(pd.to_numeric(group[f"hr_top{pct}"], errors="coerce").mean())
            row[f"baseline_timestamp_weighted_hr_top{pct}"] = float(
                pd.to_numeric(group[f"baseline_hr_top{pct}"], errors="coerce").mean()
            )
            row[f"delta_timestamp_weighted_hr_top{pct}"] = float(
                pd.to_numeric(group[f"delta_hr_top{pct}"], errors="coerce").mean()
            )
            row[f"trade_weighted_hr_top{pct}"] = _weighted_rate(group[f"hit_count_top{pct}"], group[f"selected_count_top{pct}"])
            row[f"baseline_trade_weighted_hr_top{pct}"] = _weighted_rate(
                group[f"baseline_hit_count_top{pct}"], group[f"baseline_selected_count_top{pct}"]
            )
            row[f"selected_rows_top{pct}"] = int(pd.to_numeric(group[f"selected_count_top{pct}"], errors="coerce").fillna(0).sum())
            row[f"net_correct_trades_gained_top{pct}"] = float(
                pd.to_numeric(group[f"hit_count_top{pct}"], errors="coerce").fillna(0).sum()
                - pd.to_numeric(group[f"baseline_hit_count_top{pct}"], errors="coerce").fillna(0).sum()
            )
        for metric in ("dcg", "ndcg", "average_precision", "pairwise_concordance"):
            row[f"{metric}_top30"] = float(pd.to_numeric(group[f"{metric}_top30"], errors="coerce").mean())
            row[f"baseline_{metric}_top30"] = float(pd.to_numeric(group[f"baseline_{metric}_top30"], errors="coerce").mean())
            row[f"delta_{metric}_top30"] = float(pd.to_numeric(group[f"delta_{metric}_top30"], errors="coerce").mean())
        for band in ("90_100", "80_90", "70_80"):
            row[f"timestamp_weighted_hr_{band}"] = float(pd.to_numeric(group[f"hr_{band}"], errors="coerce").mean())
            row[f"baseline_timestamp_weighted_hr_{band}"] = float(
                pd.to_numeric(group[f"baseline_hr_{band}"], errors="coerce").mean()
            )
            row[f"delta_timestamp_weighted_hr_{band}"] = float(pd.to_numeric(group[f"delta_hr_{band}"], errors="coerce").mean())
        top30_count = pd.to_numeric(group.get("selected_count_top30"), errors="coerce").fillna(0.0)
        top30_n = float(top30_count.sum())
        if top30_n > 0.0:
            row["top30_log_loss"] = float(pd.to_numeric(group["top30_log_loss_sum"], errors="coerce").fillna(0.0).sum() / top30_n)
            row["top30_baseline_on_selected_log_loss"] = float(
                pd.to_numeric(group["top30_baseline_on_selected_log_loss_sum"], errors="coerce").fillna(0.0).sum() / top30_n
            )
            row["top30_delta_log_loss_on_selected"] = row["top30_baseline_on_selected_log_loss"] - row["top30_log_loss"]
            row["top30_brier"] = float(pd.to_numeric(group["top30_brier_sum"], errors="coerce").fillna(0.0).sum() / top30_n)
            row["top30_baseline_on_selected_brier"] = float(
                pd.to_numeric(group["top30_baseline_on_selected_brier_sum"], errors="coerce").fillna(0.0).sum() / top30_n
            )
            row["top30_delta_brier_on_selected"] = row["top30_baseline_on_selected_brier"] - row["top30_brier"]
        week = group.copy()
        week_hits = week.groupby("week", sort=True)["hit_count_top30"].sum()
        week_counts = week.groupby("week", sort=True)["selected_count_top30"].sum()
        week_hr = week_hits / week_counts.replace(0, np.nan)
        row["worst_week_hr_top30"] = float(week_hr.min()) if len(week_hr) else np.nan
        row["q10_week_hr_top30"] = float(week_hr.quantile(0.10)) if len(week_hr) else np.nan
        row["week_count_top30"] = int(week_hr.notna().sum())
        row["normal_period_delta_hr_top30"] = float(
            pd.to_numeric(group.loc[group["period_type"].eq("normal_period"), "delta_hr_top30"], errors="coerce").mean()
        )
        row["bad_period_delta_hr_top30"] = float(
            pd.to_numeric(group.loc[group["period_type"].eq("bad_period"), "delta_hr_top30"], errors="coerce").mean()
        )
        row["top30_jaccard"] = float(pd.to_numeric(group["top30_jaccard"], errors="coerce").mean())
        entrants = pd.to_numeric(group["top30_entrant_count"], errors="coerce").fillna(0.0)
        removals = pd.to_numeric(group["top30_removed_count"], errors="coerce").fillna(0.0)
        row["top30_entrant_count"] = int(entrants.sum())
        row["top30_removed_count"] = int(removals.sum())
        row["top30_entrant_hit_rate"] = _weighted_rate(group["top30_entrant_hits"], group["top30_entrant_count"])
        row["top30_removed_hit_rate"] = _weighted_rate(group["top30_removed_hits"], group["top30_removed_count"])
        row["net_correct_trades_gained"] = float(pd.to_numeric(group["net_correct_trades_gained"], errors="coerce").fillna(0.0).sum())
        for key in ("mean_return", "winner_magnitude", "loser_magnitude", "lower_tail_return"):
            row[f"timestamp_weighted_top30_{key}"] = float(pd.to_numeric(group[f"top30_{key}"], errors="coerce").mean())
            row[f"timestamp_weighted_baseline_top30_{key}"] = float(pd.to_numeric(group[f"baseline_top30_{key}"], errors="coerce").mean())
            row[f"timestamp_weighted_top30_delta_{key}"] = (
                row[f"timestamp_weighted_top30_{key}"] - row[f"timestamp_weighted_baseline_top30_{key}"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _directional_episode_metrics(timestamp_metrics: pd.DataFrame, bad_episodes: set[str]) -> pd.DataFrame:
    if timestamp_metrics.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for episode in sorted(set(str(x) for x in bad_episodes)):
        subset = timestamp_metrics.loc[timestamp_metrics["episode"].astype(str).eq(episode)].copy()
        if subset.empty:
            continue
        agg = _directional_aggregate(subset)
        if not agg.empty:
            agg["heldout_episode"] = episode
            agg["period_type"] = "bad_episode"
            rows.append(agg)
    normal = timestamp_metrics.loc[~timestamp_metrics["episode"].astype(str).isin(set(str(x) for x in bad_episodes))].copy()
    if not normal.empty:
        agg = _directional_aggregate(normal)
        if not agg.empty:
            agg["heldout_episode"] = "normal_period"
            agg["period_type"] = "normal_period"
            rows.append(agg)
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()


def _directional_episode_block_confidence_intervals(
    directional_episode: pd.DataFrame,
    *,
    seed: int,
    bootstrap_rounds: int = 1000,
) -> pd.DataFrame:
    if directional_episode.empty:
        return pd.DataFrame()
    bad = directional_episode.loc[directional_episode.get("period_type", "").astype(str).eq("bad_episode")].copy()
    if bad.empty:
        return pd.DataFrame()
    metrics = [
        "delta_timestamp_weighted_hr_top10",
        "delta_timestamp_weighted_hr_top20",
        "delta_timestamp_weighted_hr_top30",
        "delta_ndcg_top30",
        "delta_average_precision_top30",
    ]
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for (head, arm, variant), group in bad.groupby(["head", "arm", "distillation_variant"], sort=True):
        for metric in metrics:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                continue
            if values.size == 1:
                lo = hi = float(values[0])
            else:
                boot = np.empty(int(bootstrap_rounds), dtype=np.float64)
                for i in range(int(bootstrap_rounds)):
                    sample = rng.choice(values, size=len(values), replace=True)
                    boot[i] = float(np.nanmean(sample))
                lo, hi = np.nanquantile(boot, [0.05, 0.95])
            rows.append(
                {
                    "head": head,
                    "arm": arm,
                    "distillation_variant": variant,
                    "metric": metric,
                    "episode_count": int(values.size),
                    "mean": float(np.nanmean(values)),
                    "median": float(np.nanmedian(values)),
                    "positive_episode_rate": float(np.mean(values > 0.0)),
                    "ci05": float(lo),
                    "ci95": float(hi),
                    "ci_method": "directional_episode_block_bootstrap",
                }
            )
    return pd.DataFrame(rows)


def _directional_selection_tuple(row: dict[str, Any] | pd.Series) -> tuple[float, float, float, float, float]:
    get = row.get if hasattr(row, "get") else lambda key, default=None: default
    return (
        float(np.nan_to_num(get("delta_timestamp_weighted_hr_top30", np.nan), nan=-1e9)),
        float(np.nan_to_num(get("delta_ndcg_top30", np.nan), nan=-1e9)),
        float(np.nan_to_num(get("delta_timestamp_weighted_hr_top20", np.nan), nan=-1e9)),
        float(np.nan_to_num(get("delta_timestamp_weighted_hr_top10", np.nan), nan=-1e9)),
        float(np.nan_to_num(get("delta_log_loss_improvement", np.nan), nan=-1e9)),
    )


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, -1, dtype=np.int16)
    mask = np.isfinite(arr)
    if int(mask.sum()) < 2:
        return out
    ranks = pd.Series(arr[mask]).rank(method="first").to_numpy(dtype=np.float64)
    pct = (ranks - 1.0) / max(float(len(ranks) - 1), 1.0)
    out[mask] = np.minimum(np.floor(pct * int(n_bins)).astype(np.int16), int(n_bins) - 1)
    return out


def _logit(p: np.ndarray) -> np.ndarray:
    arr = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(arr / (1.0 - arr))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(arr, -40.0, 40.0)))).astype(np.float32, copy=False)


def _weighted_ridge_predict(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_pred: pd.DataFrame,
    *,
    alpha: float,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    if x_train.empty or x_pred.empty:
        return np.zeros(len(x_pred), dtype=np.float32)
    train = x_train.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    pred = x_pred.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    yv = np.asarray(y_train, dtype=np.float64)
    mask = np.isfinite(yv) & np.isfinite(train).any(axis=1)
    if int(mask.sum()) < max(10, train.shape[1] + 2):
        return np.zeros(len(x_pred), dtype=np.float32)
    train = train[mask]
    yv = yv[mask]
    if sample_weight is None:
        weight = np.ones(len(yv), dtype=np.float64)
    else:
        weight = np.asarray(sample_weight, dtype=np.float64)[mask]
        weight = np.where(np.isfinite(weight) & (weight > 0.0), weight, 1.0)
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    train = np.where(np.isfinite(train), train, med)
    pred = np.where(np.isfinite(pred), pred, med)
    mean = np.average(train, axis=0, weights=weight)
    centered = train - mean
    scale = np.sqrt(np.average(centered * centered, axis=0, weights=weight))
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    train_z = (train - mean) / scale
    pred_z = (pred - mean) / scale
    xw = np.column_stack([np.ones(len(train_z)), train_z])
    xp = np.column_stack([np.ones(len(pred_z)), pred_z])
    sw = np.sqrt(weight / max(float(np.nanmean(weight)), 1e-12))
    xw = xw * sw[:, None]
    yw = yv * sw
    penalty = np.eye(xw.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    try:
        coef = np.linalg.solve(xw.T @ xw + penalty, xw.T @ yw)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(xw.T @ xw + penalty) @ (xw.T @ yw)
    out = xp @ coef
    return out.astype(np.float32, copy=False)


def _timestamp_context(canonical: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    cols = [c for c in CANONICAL_CONTEXT if c in canonical.columns]
    if not cols:
        return pd.DataFrame(index=canonical.index)
    frame = canonical.loc[:, cols].apply(pd.to_numeric, errors="coerce").copy()
    ts_key = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    frame["_timestamp"] = ts_key
    grouped = frame.groupby("_timestamp", sort=False)[cols].transform("mean")
    grouped.columns = [f"ts_mean__{c}" for c in cols]
    return _downcast_numeric(grouped)


def _fit_predict_rank_preserving_calibration(
    canonical: pd.DataFrame,
    y: np.ndarray,
    baseline_pred: np.ndarray,
    folds: list[FoldContext],
    *,
    timestamps: pd.Series,
    ridge_alpha: float = 25.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    z = _timestamp_context(canonical, timestamps)
    pred = np.full(len(y), np.nan, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce").reset_index(drop=True)
    base_logit = _logit(baseline_pred)
    for fold in folds:
        tr = np.asarray(fold.train_idx, dtype=np.int64)
        va = np.asarray(fold.valid_idx, dtype=np.int64)
        tr_mask = (y[tr] >= 0) & np.isfinite(base_logit[tr])
        va_mask = (y[va] >= 0) & np.isfinite(base_logit[va])
        tr = tr[tr_mask]
        va = va[va_mask]
        if len(tr) < 200 or len(va) < 50 or z.empty:
            pred[va] = baseline_pred[va]
            rows.append({"fold": fold.fold_id, "reason": "identity_insufficient_rows_or_context"})
            continue
        train_groups = pd.DataFrame(
            {
                "timestamp": ts.iloc[tr].to_numpy(),
                "y": y[tr].astype(np.float32),
                "base_logit": base_logit[tr],
            }
        )
        grouped = train_groups.groupby("timestamp", sort=False).agg(
            y_mean=("y", "mean"),
            base_logit_mean=("base_logit", "mean"),
            rows=("y", "size"),
        )
        if len(grouped) < 20:
            pred[va] = baseline_pred[va]
            rows.append({"fold": fold.fold_id, "reason": "identity_insufficient_timestamp_groups"})
            continue
        y_rate = np.clip(grouped["y_mean"].to_numpy(dtype=np.float64), 1e-3, 1.0 - 1e-3)
        shift_target = _logit(y_rate) - grouped["base_logit_mean"].to_numpy(dtype=np.float64)
        z_by_ts = z.assign(_timestamp=ts).groupby("_timestamp", sort=False).mean(numeric_only=True)
        train_z = z_by_ts.reindex(grouped.index)
        shift = _weighted_ridge_predict(
            train_z,
            shift_target,
            z.iloc[va].loc[:, list(z_by_ts.columns)],
            alpha=float(ridge_alpha),
            sample_weight=grouped["rows"].to_numpy(dtype=np.float64),
        )
        shift = np.clip(np.asarray(shift, dtype=np.float32), -2.5, 2.5)
        pred[va] = _sigmoid(base_logit[va] + shift)
        rows.append(
            {
                "fold": int(fold.fold_id),
                "reason": "timestamp_logit_shift_rank_preserving",
                "train_rows": int(len(tr)),
                "valid_rows": int(len(va)),
                "feature_count": int(train_z.shape[1]),
                "timestamp_groups": int(len(grouped)),
                "rank_preserving_within_timestamp": True,
            }
        )
    return pred, rows


def _fit_predict_contextual_correction(
    canonical: pd.DataFrame,
    y: np.ndarray,
    baseline_pred: np.ndarray,
    folds: list[FoldContext],
    *,
    timestamps: pd.Series,
    timestamp_alpha: float = 25.0,
    row_alpha: float = 250.0,
    correction_clip: float = 0.65,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    g_pred, g_rows = _fit_predict_rank_preserving_calibration(
        canonical,
        y,
        baseline_pred,
        folds,
        timestamps=timestamps,
        ridge_alpha=timestamp_alpha,
    )
    pred = g_pred.copy()
    rows: list[dict[str, Any]] = []
    base_logit = _logit(baseline_pred)
    g_logit = _logit(np.where(np.isfinite(g_pred), g_pred, baseline_pred))
    model_cols = [c for c in MODEL_STATE if c in canonical.columns]
    x_model = canonical.loc[:, model_cols].apply(pd.to_numeric, errors="coerce") if model_cols else pd.DataFrame(index=canonical.index)
    for fold, g_row in zip(folds, g_rows):
        tr = np.asarray(fold.train_idx, dtype=np.int64)
        va = np.asarray(fold.valid_idx, dtype=np.int64)
        tr = tr[(y[tr] >= 0) & np.isfinite(g_logit[tr])]
        va = va[(y[va] >= 0) & np.isfinite(g_logit[va])]
        if len(tr) < 200 or len(va) < 50 or x_model.empty:
            rows.append({**g_row, "reason": "timestamp_only_no_row_correction", "row_feature_count": 0})
            continue
        p = np.clip(_sigmoid(g_logit[tr]), 1e-4, 1.0 - 1e-4).astype(np.float64)
        denom = np.clip(p * (1.0 - p), 1e-3, None)
        residual = np.clip((y[tr].astype(np.float64) - p) / denom, -2.0, 2.0)
        correction = _weighted_ridge_predict(
            x_model.iloc[tr],
            residual,
            x_model.iloc[va],
            alpha=float(row_alpha),
            sample_weight=denom,
        )
        correction = np.clip(np.asarray(correction, dtype=np.float32), -float(correction_clip), float(correction_clip))
        pred[va] = _sigmoid(g_logit[va] + correction)
        rows.append(
            {
                **g_row,
                "reason": "timestamp_logit_shift_plus_regularized_model_state_delta",
                "row_feature_count": int(len(model_cols)),
                "row_correction_alpha": float(row_alpha),
                "row_correction_clip": float(correction_clip),
            }
        )
    return pred, rows


def _head_is_eligible(eligible_heads: Any, head: str) -> bool:
    raw = "" if pd.isna(eligible_heads) else str(eligible_heads)
    parts = {part.strip() for part in raw.replace("|", ",").replace(";", ",").split(",") if part.strip()}
    return "all" in parts or head in parts


def _load_episode_registry(path: str | Path, *, head: str, target_name: str) -> tuple[set[str], dict[str, Any]]:
    registry_path = Path(path) if str(path or "").strip() else DEFAULT_EPISODE_REGISTRY
    if not registry_path.exists():
        return set(), {"source": str(registry_path), "reason": "registry_missing", "fallback_allowed": True}
    df = pd.read_csv(registry_path)
    required = {
        "episode_id",
        "definition",
        "target",
        "start",
        "end",
        "severity",
        "eligible_heads",
        "reason_for_inclusion",
        "reason_for_exclusion",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Episode registry {registry_path} is missing required columns: {missing}")
    eligible = df.loc[df["eligible_heads"].map(lambda value: _head_is_eligible(value, head))].copy()
    target_values = eligible["target"].fillna("").astype(str).str.lower()
    target_ok = target_values.isin({"", "all", "diagnostic", "validation_episode_only", str(target_name).lower()})
    target_ok |= target_values.str.contains("diagnostic", regex=False)
    target_ok |= target_values.str.contains(str(target_name).lower(), regex=False)
    eligible = eligible.loc[target_ok]
    excluded = eligible["reason_for_exclusion"].fillna("").astype(str).str.strip().ne("")
    eligible = eligible.loc[~excluded]
    episodes = {
        pd.Timestamp(value).strftime("%Y-%m-%d")
        for value in eligible["episode_id"].dropna().astype(str)
        if str(value).strip()
    }
    return episodes, {
        "source": str(registry_path),
        "reason": "frozen_episode_registry",
        "registry_rows": int(len(df)),
        "eligible_rows": int(len(eligible)),
        "excluded_rows": int(excluded.sum()),
        "episodes": sorted(episodes),
        "target_name": str(target_name),
    }


def _fit_leaf_occupancy_novelty(panel: pd.DataFrame, folds: list[FoldContext]) -> pd.Series:
    source = pd.DataFrame(index=panel.index)
    if "oof_leaf_train_freq_p10" in panel.columns:
        source["low_leaf_frequency"] = -pd.to_numeric(panel["oof_leaf_train_freq_p10"], errors="coerce")
    if "oof_leaf_low_freq_fraction" in panel.columns:
        source["low_freq_fraction"] = pd.to_numeric(panel["oof_leaf_low_freq_fraction"], errors="coerce")
    if "oof_leaf_surprisal_mean" in panel.columns:
        source["leaf_surprisal"] = pd.to_numeric(panel["oof_leaf_surprisal_mean"], errors="coerce")
    if "oof_support_gap" in panel.columns:
        source["support_gap"] = pd.to_numeric(panel["oof_support_gap"], errors="coerce")
    if source.empty:
        return pd.Series(np.nan, index=panel.index, name="leaf_occupancy_novelty", dtype="float32")
    raw = source.mean(axis=1, skipna=True).to_numpy(dtype=np.float32)
    out = np.full(len(panel), np.nan, dtype=np.float32)
    for fold in folds:
        train = np.asarray(fold.train_idx, dtype=np.int64)
        valid = np.asarray(fold.valid_idx, dtype=np.int64)
        train_values = raw[train]
        finite_train = train_values[np.isfinite(train_values)]
        if finite_train.size < 20:
            continue
        lo = float(np.nanquantile(finite_train, 0.05))
        hi = float(np.nanquantile(finite_train, 0.95))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue
        for idx in (train, valid):
            vals = np.clip((raw[idx] - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
            out[idx] = vals.astype(np.float32, copy=False)
    return pd.Series(out, index=panel.index, name="leaf_occupancy_novelty", dtype="float32")


def _interaction_features(panel: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=canonical.index)
    for left, right in INTERACTION_PAIRS:
        l = pd.to_numeric(canonical.get(left), errors="coerce").to_numpy(dtype=np.float32)
        r = pd.to_numeric(canonical.get(right), errors="coerce").to_numpy(dtype=np.float32)
        out[f"{left}__x__{right}"] = (l * r).astype(np.float32, copy=False)
    support = pd.to_numeric(canonical.get("prediction_support_quality"), errors="coerce").to_numpy(dtype=np.float32)
    for score_col in ("oof_base_clf", "oof_meta_clf", "oof_pred", "oof_rank_pct"):
        if score_col in panel.columns:
            score = pd.to_numeric(panel[score_col], errors="coerce").to_numpy(dtype=np.float32)
            out[f"{score_col}__x__prediction_support_quality"] = (score * support).astype(np.float32, copy=False)
    stress = pd.DataFrame(index=canonical.index)
    for col in ("tail_volatility_stress", "liquidity_participation_stress", "leverage_funding_crowding"):
        stress[col] = pd.to_numeric(canonical.get(col), errors="coerce")
    stress_score = stress.mean(axis=1, skipna=True).to_numpy(dtype=np.float32)
    for path_col in ("oof_score_path_std", "oof_rank_path_std", "oof_score_path_volatility", "oof_rolling_cluster_stability"):
        if path_col in panel.columns:
            path = pd.to_numeric(panel[path_col], errors="coerce").to_numpy(dtype=np.float32)
            out[f"{path_col}__x__market_state_stress"] = (path * stress_score).astype(np.float32, copy=False)
    return _downcast_numeric(out)


def _arm_frames(panel: pd.DataFrame, current_x: pd.DataFrame, canonical: pd.DataFrame) -> dict[str, pd.DataFrame | None]:
    model_state = canonical.loc[:, list(MODEL_STATE)]
    market_state = canonical.loc[:, list(MARKET_STATE)]
    both = pd.concat([model_state, market_state], axis=1, copy=False)
    interactions = _interaction_features(panel, canonical)
    return {
        ARM_A: None,
        ARM_B: pd.concat([current_x, model_state], axis=1, copy=False),
        ARM_C: pd.concat([current_x, market_state], axis=1, copy=False),
        ARM_D: pd.concat([current_x, both], axis=1, copy=False),
        ARM_E: pd.concat([current_x, both, interactions], axis=1, copy=False),
    }


def _reliability_weights(canonical: pd.DataFrame, variant: str, lambda0: float) -> np.ndarray:
    n = len(canonical)
    if variant == "hard_label_only":
        return np.zeros(n, dtype=np.float32)
    if variant == "lower_global_distillation_weight":
        return np.full(n, float(lambda0) * 0.35, dtype=np.float32)
    if variant == "current_self_distillation":
        return np.full(n, float(lambda0), dtype=np.float32)
    support_quality = pd.to_numeric(canonical.get("prediction_support_quality"), errors="coerce").to_numpy(dtype=np.float32)
    novelty = pd.to_numeric(canonical.get("leaf_occupancy_novelty"), errors="coerce").to_numpy(dtype=np.float32)
    path_instability = pd.to_numeric(canonical.get("prediction_path_instability"), errors="coerce").to_numpy(dtype=np.float32)
    support = 1.0 - np.nan_to_num(novelty, nan=0.5)
    reliability = 0.5 * np.nan_to_num(support_quality, nan=0.5) + 0.5 * (1.0 - np.nan_to_num(path_instability, nan=0.5))
    market = canonical.loc[:, [c for c in MARKET_STATE if c in canonical.columns]].apply(pd.to_numeric, errors="coerce")
    stress = market.mean(axis=1, skipna=True).to_numpy(dtype=np.float32) if not market.empty else np.full(n, 0.5)
    stress_reliability = 1.0 - np.clip(np.nan_to_num(stress, nan=0.5), 0.0, 1.0)
    if variant == "support_aware_distillation":
        weight = support
    elif variant == "context_aware_distillation":
        weight = reliability * stress_reliability
    else:
        weight = reliability * support * stress_reliability
    return np.clip(float(lambda0) * weight, 0.0, float(lambda0)).astype(np.float32, copy=False)


def _fit_predict_classifier(
    x: pd.DataFrame,
    y: np.ndarray,
    folds: list[FoldContext],
    *,
    timestamps: pd.Series,
    seed: int,
    max_train_rows: int,
    max_depth: int = 3,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if lgb is None:
        raise RuntimeError("lightgbm is required")
    x = x.replace([np.inf, -np.inf], np.nan)
    keep_cols = [c for c in x.columns if pd.to_numeric(x[c], errors="coerce").notna().mean() > 0.02]
    if not keep_cols:
        return np.full(len(y), np.nan, dtype=np.float32), [{"reason": "empty_matrix", "feature_count": 0}]
    x_prepared = _prepare_model_matrix(x.loc[:, keep_cols])
    pred = np.full(len(y), np.nan, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for fold in folds:
        tr = np.asarray(fold.train_idx, dtype=np.int64)
        va = np.asarray(fold.valid_idx, dtype=np.int64)
        tr = tr[y[tr] >= 0]
        va = va[y[va] >= 0]
        if len(tr) < 200 or len(va) < 50 or len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            rows.append({"fold": fold.fold_id, "reason": "insufficient_rows_or_classes"})
            continue
        tr_fit = _period_stratified_train_sample(
            timestamps=timestamps.reset_index(drop=True),
            y=np.maximum(y, 0),
            train_idx=tr,
            max_rows=int(max_train_rows),
            seed=int(seed + fold.fold_id * 17),
        )
        fit_weight = None
        if sample_weight is not None:
            fit_weight = np.asarray(sample_weight, dtype=np.float32)[tr_fit]
            fit_weight = np.where(np.isfinite(fit_weight) & (fit_weight > 0.0), fit_weight, 1.0)
        min_child = max(50, int(math.ceil(0.025 * len(tr_fit))))
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=350,
            learning_rate=0.035,
            max_depth=int(max_depth),
            num_leaves=max(4, min(16, 2 ** int(max_depth))),
            min_child_samples=min_child,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=int(seed + fold.fold_id * 1009),
            n_jobs=max(1, min(6, os.cpu_count() or 2)),
            verbosity=-1,
        )
        callbacks = [lgb.early_stopping(40, verbose=False)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(
                x_prepared.iloc[tr_fit],
                y[tr_fit],
                sample_weight=fit_weight,
                eval_set=[(x_prepared.iloc[va], y[va])],
                eval_metric="binary_logloss",
                callbacks=callbacks,
            )
        pred[va] = clf.predict_proba(x_prepared.iloc[va])[:, 1].astype(np.float32, copy=False)
        rows.append(
            {
                "fold": int(fold.fold_id),
                "reason": "",
                "train_rows": int(len(tr_fit)),
                "valid_rows": int(len(va)),
                "feature_count": int(len(keep_cols)),
                "best_iteration": int(getattr(clf, "best_iteration_", 0) or 0),
                "valid_auc": _safe_auc(y[va].astype(np.float32), pred[va]),
                "sample_weight_mean": float(np.nanmean(fit_weight)) if fit_weight is not None else 1.0,
            }
        )
    return pred, rows


def _fit_predict_distilled(
    x: pd.DataFrame,
    y: np.ndarray,
    teacher: np.ndarray,
    canonical: pd.DataFrame,
    folds: list[FoldContext],
    *,
    timestamps: pd.Series,
    seed: int,
    max_train_rows: int,
    lambda0: float,
    variant: str,
    max_depth: int = 3,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if variant == "hard_label_only":
        return _fit_predict_classifier(
            x,
            y,
            folds,
            timestamps=timestamps,
            seed=seed,
            max_train_rows=max_train_rows,
            max_depth=max_depth,
        )
    if lgb is None:
        raise RuntimeError("lightgbm is required")
    x = x.replace([np.inf, -np.inf], np.nan)
    keep_cols = [c for c in x.columns if pd.to_numeric(x[c], errors="coerce").notna().mean() > 0.02]
    if not keep_cols:
        return np.full(len(y), np.nan, dtype=np.float32), [{"reason": "empty_matrix", "feature_count": 0}]
    x_prepared = _prepare_model_matrix(x.loc[:, keep_cols])
    lambda_i = _reliability_weights(canonical, variant, lambda0=float(lambda0))
    teacher = np.clip(np.asarray(teacher, dtype=np.float32), 1e-4, 1.0 - 1e-4)
    y_float = np.maximum(y, 0).astype(np.float32, copy=False)
    soft_y = ((y_float + lambda_i * teacher) / (1.0 + lambda_i)).astype(np.float32, copy=False)
    pred = np.full(len(y), np.nan, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for fold in folds:
        tr = np.asarray(fold.train_idx, dtype=np.int64)
        va = np.asarray(fold.valid_idx, dtype=np.int64)
        tr = tr[(y[tr] >= 0) & np.isfinite(soft_y[tr])]
        va = va[y[va] >= 0]
        if len(tr) < 200 or len(va) < 50 or len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            rows.append({"fold": fold.fold_id, "reason": "insufficient_rows_or_classes"})
            continue
        tr_fit = _period_stratified_train_sample(
            timestamps=timestamps.reset_index(drop=True),
            y=np.maximum(y, 0),
            train_idx=tr,
            max_rows=int(max_train_rows),
            seed=int(seed + fold.fold_id * 31),
        )
        min_child = max(50, int(math.ceil(0.025 * len(tr_fit))))
        reg = lgb.LGBMRegressor(
            objective="cross_entropy",
            n_estimators=350,
            learning_rate=0.035,
            max_depth=int(max_depth),
            num_leaves=max(4, min(16, 2 ** int(max_depth))),
            min_child_samples=min_child,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=int(seed + fold.fold_id * 1009),
            n_jobs=max(1, min(6, os.cpu_count() or 2)),
            verbosity=-1,
        )
        callbacks = [lgb.early_stopping(40, verbose=False)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(
                x_prepared.iloc[tr_fit],
                soft_y[tr_fit],
                eval_set=[(x_prepared.iloc[va], y_float[va])],
                eval_metric="binary_logloss",
                callbacks=callbacks,
            )
        pred[va] = np.clip(reg.predict(x_prepared.iloc[va]), 1e-6, 1.0 - 1e-6).astype(np.float32, copy=False)
        rows.append(
            {
                "fold": int(fold.fold_id),
                "reason": "",
                "train_rows": int(len(tr_fit)),
                "valid_rows": int(len(va)),
                "feature_count": int(len(keep_cols)),
                "best_iteration": int(getattr(reg, "best_iteration_", 0) or 0),
                "valid_auc": _safe_auc(y[va].astype(np.float32), pred[va]),
                "mean_distillation_lambda": float(np.nanmean(lambda_i[tr_fit])),
            }
        )
    return pred, rows


def _cell_metrics(
    *,
    head: str,
    arm: str,
    variant: str,
    family: str,
    cells: pd.Series,
    y: np.ndarray,
    pred: np.ndarray,
    returns: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cells = cells.reset_index(drop=True).astype(str)
    for cell, idx in cells.groupby(cells).groups.items():
        ids = np.asarray(list(idx), dtype=np.int64)
        mask = (y[ids] >= 0) & np.isfinite(pred[ids])
        ids = ids[mask]
        if len(ids) < 30:
            continue
        yy = y[ids]
        pp = np.clip(pred[ids], 1e-6, 1.0 - 1e-6)
        rr = returns[ids]
        ret_metrics = _return_metrics(rr)
        rows.append(
            {
                "head": head,
                "arm": arm,
                "distillation_variant": variant,
                "cell_family": family,
                "cell": str(cell),
                "trade_count": int(len(ids)),
                "label_hit_rate": float(np.mean(yy)),
                "meta_prediction": float(np.mean(pp)),
                "log_loss": _safe_log_loss(yy, pp),
                "brier": _safe_brier(yy, pp),
                "mean_return": ret_metrics["mean_return"],
                "winner_magnitude": ret_metrics["winner_magnitude"],
                "loser_magnitude": ret_metrics["loser_magnitude"],
                "lower_tail_return": ret_metrics["lower_tail_return"],
            }
        )
    return rows


def _build_cell_rows(
    *,
    head: str,
    arm: str,
    variant: str,
    panel: pd.DataFrame,
    canonical: pd.DataFrame,
    y: np.ndarray,
    pred: np.ndarray,
    returns: np.ndarray,
    bad_episodes: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    episodes = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce").dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    rows.extend(
        _cell_metrics(
            head=head,
            arm=arm,
            variant=variant,
            family="period_type",
            cells=episodes.map(lambda x: "bad_period" if str(x) in bad_episodes else "normal_period"),
            y=y,
            pred=pred,
            returns=returns,
        )
    )
    support = pd.to_numeric(canonical.get("prediction_support_quality"), errors="coerce").to_numpy(dtype=np.float32)
    market = canonical.loc[:, [c for c in MARKET_STATE if c in canonical.columns]].apply(pd.to_numeric, errors="coerce")
    market_score = market.mean(axis=1, skipna=True).to_numpy(dtype=np.float32) if not market.empty else np.full(len(panel), np.nan)
    base_score = pd.to_numeric(panel.get("oof_rank_pct"), errors="coerce").to_numpy(dtype=np.float32)
    for family, values in (
        ("support_quality_decile", support),
        ("market_state_decile", market_score),
        ("base_score_decile", base_score),
    ):
        bins = _quantile_bins(values, 10)
        rows.extend(
            _cell_metrics(
                head=head,
                arm=arm,
                variant=variant,
                family=family,
                cells=pd.Series([f"{family}_{int(b) + 1}" if b >= 0 else "missing" for b in bins]),
                y=y,
                pred=pred,
                returns=returns,
            )
        )
    support_tertile = _quantile_bins(support, 3)
    market_tertile = _quantile_bins(market_score, 3)
    cells = pd.Series(
        [
            f"support_{int(s) + 1}__market_{int(m) + 1}" if s >= 0 and m >= 0 else "missing"
            for s, m in zip(support_tertile, market_tertile)
        ]
    )
    rows.extend(
        _cell_metrics(
            head=head,
            arm=arm,
            variant=variant,
            family="support_x_market_state_cell",
            cells=cells,
            y=y,
            pred=pred,
            returns=returns,
        )
    )
    return rows


def _episode_labels(panel: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(panel["timestamp"], utc=True, errors="coerce").dt.to_period("W").dt.start_time.dt.strftime(
        "%Y-%m-%d"
    )


def _regime_balance_weights(panel: pd.DataFrame, bad_episodes: set[str]) -> np.ndarray | None:
    if not bad_episodes:
        return None
    bad = _episode_labels(panel).isin(set(str(x) for x in bad_episodes)).to_numpy(dtype=bool)
    n_bad = int(bad.sum())
    n_normal = int((~bad).sum())
    if n_bad < 50 or n_normal < 50:
        return None
    n = len(bad)
    weights = np.ones(n, dtype=np.float32)
    weights[bad] = 0.5 * float(n) / float(n_bad)
    weights[~bad] = 0.5 * float(n) / float(n_normal)
    return np.clip(weights, 0.25, 8.0).astype(np.float32, copy=False)


def _gradient_conflict_diagnostics(
    *,
    head: str,
    panel: pd.DataFrame,
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    bad_episodes: set[str],
    n_score_bins: int = 10,
) -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame()
    episodes = _episode_labels(panel)
    regimes = episodes.where(episodes.isin(set(str(x) for x in bad_episodes)), other="normal_period").reset_index(drop=True)
    base_score = None
    if "oof_rank_pct" in panel.columns:
        base_score = pd.to_numeric(panel["oof_rank_pct"], errors="coerce").to_numpy(dtype=np.float32)
    if base_score is None or not np.isfinite(base_score).any():
        base_score = next(iter(predictions.values()))
    score_bins = _quantile_bins(base_score, int(n_score_bins))
    rows: list[dict[str, Any]] = []
    y_arr = np.asarray(y, dtype=np.float32)
    for arm, pred_values in predictions.items():
        pred = np.asarray(pred_values, dtype=np.float32)
        base_mask = (y_arr >= 0) & np.isfinite(pred) & (score_bins >= 0)
        if int(base_mask.sum()) < 50:
            continue
        for score_bin in sorted(set(score_bins[base_mask].astype(int))):
            region_mask = base_mask & (score_bins == score_bin)
            if int(region_mask.sum()) < 20:
                continue
            region_rows: list[dict[str, Any]] = []
            total_gradient = 0.0
            total_abs_gradient = 0.0
            regime_frame = pd.DataFrame(
                {
                    "row_id": np.flatnonzero(region_mask).astype(np.int64),
                    "regime": regimes.loc[region_mask].to_numpy(),
                }
            )
            for regime, regime_group in regime_frame.groupby("regime", sort=True):
                ids = regime_group["row_id"].to_numpy(dtype=np.int64)
                pp = np.clip(pred[ids].astype(np.float64), 1e-6, 1.0 - 1e-6)
                yy = y_arr[ids].astype(np.float64)
                gradient_sum = float(np.sum(pp - yy))
                hessian_sum = float(np.sum(pp * (1.0 - pp)))
                optimal_update = float(-gradient_sum / max(hessian_sum, 1e-9))
                total_gradient += gradient_sum
                total_abs_gradient += abs(gradient_sum)
                region_rows.append(
                    {
                        "head": head,
                        "arm": arm,
                        "score_region": f"score_decile_{int(score_bin) + 1}",
                        "leaf": f"score_decile_{int(score_bin) + 1}",
                        "regime": str(regime),
                        "support": int(len(ids)),
                        "gradient_sum": gradient_sum,
                        "hessian_sum": hessian_sum,
                        "optimal_update": optimal_update,
                        "update_sign": int(np.sign(optimal_update)),
                    }
                )
            cancellation_score = 1.0 - abs(total_gradient) / (total_abs_gradient + 1e-12)
            for row in region_rows:
                row["region_support"] = int(region_mask.sum())
                row["region_total_gradient_sum"] = total_gradient
                row["region_abs_gradient_sum"] = total_abs_gradient
                row["cancellation_score"] = float(np.clip(cancellation_score, 0.0, 1.0))
                row["diagnostic_type"] = "score_region_gradient_conflict"
                rows.append(row)
    return pd.DataFrame(rows)


def _episode_block_confidence_intervals(
    leave_one: pd.DataFrame,
    *,
    seed: int = 23,
    bootstrap_rounds: int = 400,
) -> pd.DataFrame:
    if leave_one.empty or "heldout_episode" not in leave_one.columns:
        return pd.DataFrame()
    df = leave_one.loc[leave_one.get("arm", pd.Series(dtype=str)).astype(str).ne("__summary__")].copy()
    if df.empty:
        return pd.DataFrame()
    metrics = [
        "delta_log_loss_improvement",
        "delta_brier_improvement",
        "top10_delta_mean_return",
        "top10_delta_winner_magnitude",
        "top10_delta_lower_tail_return",
    ]
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, Any]] = []
    for (head, arm, variant), group in df.groupby(["head", "arm", "distillation_variant"], sort=True):
        episodes = sorted(set(group["heldout_episode"].dropna().astype(str)))
        if not episodes:
            continue
        by_episode = group.groupby("heldout_episode", sort=True).mean(numeric_only=True)
        for metric in metrics:
            if metric not in by_episode.columns:
                continue
            values = pd.to_numeric(by_episode[metric], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if values.size == 0:
                continue
            if values.size == 1:
                lo = hi = float(values[0])
            else:
                samples = np.empty(int(bootstrap_rounds), dtype=np.float64)
                for i in range(int(bootstrap_rounds)):
                    draw = rng.choice(values, size=values.size, replace=True)
                    samples[i] = float(np.nanmean(draw))
                lo = float(np.nanquantile(samples, 0.05))
                hi = float(np.nanquantile(samples, 0.95))
            rows.append(
                {
                    "head": head,
                    "arm": arm,
                    "distillation_variant": variant,
                    "metric": metric,
                    "episode_count": int(values.size),
                    "mean": float(np.nanmean(values)),
                    "median": float(np.nanmedian(values)),
                    "positive_episode_rate": float(np.mean(values > 0.0)),
                    "ci05": lo,
                    "ci95": hi,
                    "ci_method": "episode_block_bootstrap",
                }
            )
    return pd.DataFrame(rows)


def _gradient_conflict_summary(gradient_conflict: pd.DataFrame) -> pd.DataFrame:
    if gradient_conflict.empty:
        return pd.DataFrame()
    required = {"head", "arm", "score_region", "region_support", "cancellation_score"}
    if not required <= set(gradient_conflict.columns):
        return pd.DataFrame()
    regions = gradient_conflict.drop_duplicates(["head", "arm", "score_region"]).copy()
    regions["region_support"] = pd.to_numeric(regions["region_support"], errors="coerce").fillna(0.0)
    regions["cancellation_score"] = pd.to_numeric(regions["cancellation_score"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for (head, arm), group in regions.groupby(["head", "arm"], sort=True):
        support = group["region_support"].to_numpy(dtype=np.float64)
        conflict = group["cancellation_score"].to_numpy(dtype=np.float64)
        finite = np.isfinite(conflict) & np.isfinite(support) & (support > 0)
        if not finite.any():
            continue
        support = support[finite]
        conflict = conflict[finite]
        total = float(np.sum(support))
        high = conflict > 0.8
        rows.append(
            {
                "head": head,
                "arm": arm,
                "gradient_conflict_max": float(np.max(conflict)),
                "gradient_conflict_weighted": float(np.sum(support * conflict) / max(total, 1e-12)),
                "gradient_conflict_high_row_fraction": float(np.sum(support[high]) / max(total, 1e-12)),
                "gradient_conflict_region_count": int(len(conflict)),
                "gradient_conflict_total_region_rows": int(total),
            }
        )
    return pd.DataFrame(rows)


def _promotion_table(
    summary: pd.DataFrame,
    period_conflict: pd.DataFrame,
    episode_ci: pd.DataFrame,
    gradient_conflict: pd.DataFrame,
    oracle_specialist: pd.DataFrame,
    directional: pd.DataFrame | None = None,
    directional_episode_ci: pd.DataFrame | None = None,
    *,
    winner_epsilon: float,
    lower_tail_tolerance: float,
    directional_hr10_tolerance: float = 0.0,
    directional_normal_tolerance: float = 0.0,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    table = summary.copy()
    directional = directional if directional is not None else pd.DataFrame()
    directional_episode_ci = directional_episode_ci if directional_episode_ci is not None else pd.DataFrame()
    if not directional.empty:
        table = table.merge(
            directional,
            on=["head", "arm", "distillation_variant"],
            how="left",
            suffixes=("", "_directional"),
        )
    if not period_conflict.empty:
        pc = period_conflict[
            [
                c
                for c in [
                    "head",
                    "arm",
                    "bad_period_delta_log_loss_improvement",
                    "normal_period_delta_log_loss_improvement",
                    "period_objective_conflict",
                ]
                if c in period_conflict.columns
            ]
        ].drop_duplicates(["head", "arm"])
        table = table.merge(pc, on=["head", "arm"], how="left")
    if not episode_ci.empty:
        metrics = [
            "delta_log_loss_improvement",
            "delta_brier_improvement",
            "top10_delta_mean_return",
            "top10_delta_winner_magnitude",
            "top10_delta_lower_tail_return",
        ]
        ci = episode_ci.loc[episode_ci["metric"].astype(str).isin(metrics)].copy()
        if not ci.empty:
            pivots = []
            for value_col, prefix in (
                ("median", "episode_median"),
                ("positive_episode_rate", "episode_positive_rate"),
                ("ci05", "episode_ci05"),
                ("ci95", "episode_ci95"),
                ("episode_count", "episode_count"),
            ):
                pivot = ci.pivot_table(
                    index=["head", "arm", "distillation_variant"],
                    columns="metric",
                    values=value_col,
                    aggfunc="first",
                )
                pivot.columns = [f"{prefix}_{col}" for col in pivot.columns]
                pivots.append(pivot.reset_index())
            ci_wide = pivots[0]
            for pivot in pivots[1:]:
                ci_wide = ci_wide.merge(pivot, on=["head", "arm", "distillation_variant"], how="outer")
            table = table.merge(ci_wide, on=["head", "arm", "distillation_variant"], how="left")
    if not directional_episode_ci.empty:
        metrics = [
            "delta_timestamp_weighted_hr_top10",
            "delta_timestamp_weighted_hr_top20",
            "delta_timestamp_weighted_hr_top30",
            "delta_ndcg_top30",
            "delta_average_precision_top30",
        ]
        ci = directional_episode_ci.loc[directional_episode_ci["metric"].astype(str).isin(metrics)].copy()
        if not ci.empty:
            pivots = []
            for value_col, prefix in (
                ("median", "directional_episode_median"),
                ("positive_episode_rate", "directional_episode_positive_rate"),
                ("ci05", "directional_episode_ci05"),
                ("ci95", "directional_episode_ci95"),
                ("episode_count", "directional_episode_count"),
            ):
                pivot = ci.pivot_table(
                    index=["head", "arm", "distillation_variant"],
                    columns="metric",
                    values=value_col,
                    aggfunc="first",
                )
                pivot.columns = [f"{prefix}_{col}" for col in pivot.columns]
                pivots.append(pivot.reset_index())
            ci_wide = pivots[0]
            for pivot in pivots[1:]:
                ci_wide = ci_wide.merge(pivot, on=["head", "arm", "distillation_variant"], how="outer")
            table = table.merge(ci_wide, on=["head", "arm", "distillation_variant"], how="left")
    gc = _gradient_conflict_summary(gradient_conflict)
    if not gc.empty:
        table = table.merge(gc, on=["head", "arm"], how="left")
    if not oracle_specialist.empty:
        evaluated = oracle_specialist.loc[oracle_specialist.get("status", pd.Series(dtype=str)).astype(str).eq("evaluated")].copy()
        if not evaluated.empty:
            oracle = (
                evaluated.loc[evaluated["benchmark_model"].astype(str).eq(ARM_J)]
                .groupby("head", as_index=False)["delta_log_loss_improvement"]
                .mean()
                .rename(columns={"delta_log_loss_improvement": "oracle_specialist_mean_delta_log_loss"})
            )
            model = (
                evaluated.groupby(["head", "benchmark_model"], as_index=False)["delta_log_loss_improvement"]
                .mean()
                .rename(
                    columns={
                        "benchmark_model": "arm",
                        "delta_log_loss_improvement": "oracle_benchmark_arm_mean_delta_log_loss",
                    }
                )
            )
            table = table.merge(oracle, on="head", how="left").merge(model, on=["head", "arm"], how="left")
            table["oracle_specialist_gap_log_loss"] = (
                table["oracle_specialist_mean_delta_log_loss"] - table["oracle_benchmark_arm_mean_delta_log_loss"]
            )
    table["equal_exposure_top10_mean_return_change"] = pd.to_numeric(
        table.get("top10_delta_mean_return", pd.Series(np.nan, index=table.index)), errors="coerce"
    )
    table["passes_economic_constraints"] = (
        pd.to_numeric(table.get("top10_delta_mean_return", pd.Series(np.nan, index=table.index)), errors="coerce").fillna(-np.inf)
        >= 0.0
    ) & (
        pd.to_numeric(table.get("top10_delta_winner_magnitude", pd.Series(np.nan, index=table.index)), errors="coerce").fillna(-np.inf)
        >= -float(winner_epsilon)
    ) & (
        pd.to_numeric(table.get("top10_delta_lower_tail_return", pd.Series(np.nan, index=table.index)), errors="coerce").fillna(-np.inf)
        >= -float(lower_tail_tolerance)
    )
    table["passes_episode_label_constraints"] = (
        pd.to_numeric(
            table.get("episode_median_delta_log_loss_improvement", pd.Series(np.nan, index=table.index)), errors="coerce"
        ).fillna(-np.inf)
        > 0.0
    ) & (
        pd.to_numeric(
            table.get("episode_positive_rate_delta_brier_improvement", pd.Series(np.nan, index=table.index)), errors="coerce"
        ).fillna(0.0)
        >= 0.5
    )
    directional_episode_count = pd.to_numeric(
        table.get("directional_episode_count_delta_timestamp_weighted_hr_top30", pd.Series(np.nan, index=table.index)),
        errors="coerce",
    ).fillna(0.0)
    directional_episode_median_hr30 = pd.to_numeric(
        table.get("directional_episode_median_delta_timestamp_weighted_hr_top30", pd.Series(np.nan, index=table.index)),
        errors="coerce",
    )
    directional_episode_positive_hr30 = pd.to_numeric(
        table.get("directional_episode_positive_rate_delta_timestamp_weighted_hr_top30", pd.Series(np.nan, index=table.index)),
        errors="coerce",
    )
    delta_hr30 = pd.to_numeric(table.get("delta_timestamp_weighted_hr_top30", pd.Series(np.nan, index=table.index)), errors="coerce")
    delta_hr20 = pd.to_numeric(table.get("delta_timestamp_weighted_hr_top20", pd.Series(np.nan, index=table.index)), errors="coerce")
    delta_hr10 = pd.to_numeric(table.get("delta_timestamp_weighted_hr_top10", pd.Series(np.nan, index=table.index)), errors="coerce")
    delta_ndcg = pd.to_numeric(table.get("delta_ndcg_top30", pd.Series(np.nan, index=table.index)), errors="coerce")
    normal_delta_hr30 = pd.to_numeric(table.get("normal_period_delta_hr_top30", pd.Series(np.nan, index=table.index)), errors="coerce")
    table["passes_directional_pooled_constraints"] = (
        delta_hr30.fillna(-np.inf).gt(0.0)
        & delta_ndcg.fillna(-np.inf).ge(0.0)
        & delta_hr10.fillna(-np.inf).ge(-float(directional_hr10_tolerance))
        & delta_hr20.fillna(-np.inf).ge(-float(directional_hr10_tolerance))
        & normal_delta_hr30.fillna(0.0).ge(-float(directional_normal_tolerance))
    )
    table["passes_directional_episode_constraints"] = (
        directional_episode_count.eq(0.0)
        | (
            directional_episode_median_hr30.fillna(-np.inf).gt(0.0)
            & directional_episode_positive_hr30.fillna(0.0).ge(0.75)
        )
    )
    table["directional_promotion_candidate"] = (
        table["passes_directional_pooled_constraints"]
        & table["passes_directional_episode_constraints"]
        & table["arm"].astype(str).ne(ARM_A)
    )
    table["promotion_candidate"] = (
        table["directional_promotion_candidate"]
    )
    episode_rank = directional_episode_median_hr30.fillna(delta_hr30)
    table["selection_rank_score"] = np.where(
        table["promotion_candidate"],
        episode_rank.fillna(0.0) * 1_000_000.0
        + delta_hr30.fillna(0.0) * 100_000.0
        + delta_ndcg.fillna(0.0) * 10_000.0
        + delta_hr20.fillna(0.0) * 1_000.0
        + pd.to_numeric(table.get("top30_delta_log_loss_on_selected", pd.Series(0.0, index=table.index)), errors="coerce").fillna(0.0),
        -np.inf,
    )
    for col in (
        "directional_episode_median_delta_timestamp_weighted_hr_top30",
        "delta_timestamp_weighted_hr_top30",
        "delta_ndcg_top30",
    ):
        if col not in table.columns:
            table[col] = np.nan
    return table.sort_values(
        [
            "head",
            "promotion_candidate",
            "directional_episode_median_delta_timestamp_weighted_hr_top30",
            "delta_timestamp_weighted_hr_top30",
            "delta_ndcg_top30",
            "selection_rank_score",
        ],
        ascending=[True, False, False, False, False, False],
    )


def _bad_episode_context_diagnostics(
    *,
    head: str,
    panel: pd.DataFrame,
    canonical: pd.DataFrame,
    y: np.ndarray,
    baseline_pred: np.ndarray,
    returns: np.ndarray,
    predictions: dict[str, np.ndarray],
    bad_episodes: set[str],
    selected_arms: list[str],
) -> pd.DataFrame:
    if not bad_episodes:
        return pd.DataFrame()
    episodes = _episode_labels(panel)
    market_cols = [c for c in MARKET_STATE if c in canonical.columns]
    model_cols = [c for c in MODEL_STATE if c in canonical.columns]
    rows: list[dict[str, Any]] = []
    y_arr = np.asarray(y, dtype=np.float32)
    returns_arr = np.asarray(returns, dtype=np.float32)
    base = np.asarray(baseline_pred, dtype=np.float32)
    for arm in selected_arms:
        pred = np.asarray(predictions.get(arm, np.full(len(panel), np.nan)), dtype=np.float32)
        for episode in sorted(bad_episodes):
            mask = episodes.eq(str(episode)).to_numpy(dtype=bool) & (y_arr >= 0)
            if int(mask.sum()) == 0:
                continue
            ids = np.flatnonzero(mask)
            metrics = _overall_metrics(
                head=head,
                arm=arm,
                variant="episode_context_diagnostic",
                y=y_arr[ids].astype(np.int8),
                pred=pred[ids],
                baseline_pred=base[ids],
                returns=returns_arr[ids],
            )
            market = canonical.loc[ids, market_cols].apply(pd.to_numeric, errors="coerce") if market_cols else pd.DataFrame(index=ids)
            model = canonical.loc[ids, model_cols].apply(pd.to_numeric, errors="coerce") if model_cols else pd.DataFrame(index=ids)
            score = pd.to_numeric(panel.get("oof_rank_pct"), errors="coerce").to_numpy(dtype=np.float32) if "oof_rank_pct" in panel else base
            rows.append(
                {
                    "head": head,
                    "heldout_episode": str(episode),
                    "arm": arm,
                    "rows": int(len(ids)),
                    "label_rate": float(np.nanmean(y_arr[ids])),
                    "delta_log_loss_improvement": metrics.get("delta_log_loss_improvement", np.nan),
                    "delta_brier_improvement": metrics.get("delta_brier_improvement", np.nan),
                    "top10_delta_mean_return": metrics.get("top10_delta_mean_return", np.nan),
                    "top10_delta_winner_magnitude": metrics.get("top10_delta_winner_magnitude", np.nan),
                    "top10_delta_lower_tail_return": metrics.get("top10_delta_lower_tail_return", np.nan),
                    "score_mean": float(np.nanmean(score[ids])),
                    "score_q10": float(np.nanquantile(score[ids], 0.10)) if len(ids) >= 10 else np.nan,
                    "score_q50": float(np.nanquantile(score[ids], 0.50)) if len(ids) >= 10 else np.nan,
                    "score_q90": float(np.nanquantile(score[ids], 0.90)) if len(ids) >= 10 else np.nan,
                    "market_context_mean": float(np.nanmean(market.to_numpy(dtype=np.float32))) if not market.empty else np.nan,
                    "model_context_mean": float(np.nanmean(model.to_numpy(dtype=np.float32))) if not model.empty else np.nan,
                    "context_missing_fraction": float(canonical.loc[ids].isna().mean().mean()),
                    "classification_hint": _classify_episode_failure(metrics, len(ids), canonical.loc[ids]),
                }
            )
    return pd.DataFrame(rows)


def _classify_episode_failure(metrics: dict[str, Any], rows: int, context_slice: pd.DataFrame) -> str:
    log_delta = float(metrics.get("delta_log_loss_improvement", np.nan))
    tail_delta = float(metrics.get("top10_delta_lower_tail_return", np.nan))
    mean_delta = float(metrics.get("top10_delta_mean_return", np.nan))
    missing = float(context_slice.isna().mean().mean()) if not context_slice.empty else 1.0
    if rows < 200:
        return "sampling_uncertainty"
    if missing > 0.35:
        return "context_coverage_failure"
    if np.isfinite(log_delta) and log_delta < 0 and np.isfinite(mean_delta) and mean_delta >= 0:
        return "label_economics_mismatch"
    if np.isfinite(log_delta) and log_delta < 0:
        return "model_capacity_or_distinct_regime"
    if np.isfinite(tail_delta) and tail_delta < 0 and np.isfinite(mean_delta) and mean_delta >= 0:
        return "high_dispersion_payoff_structure"
    return "no_material_failure"


def _leave_one_bad_episode_rows(
    *,
    head: str,
    panel: pd.DataFrame,
    arms: dict[str, pd.DataFrame | None],
    best_arm: str,
    canonical: pd.DataFrame,
    y: np.ndarray,
    baseline_pred: np.ndarray,
    returns: np.ndarray,
    bad_episodes: set[str],
    seed: int,
    max_train_rows: int,
    embargo_hours: int,
    distillation_lambda: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not bad_episodes:
        return [
            {
                "head": head,
                "arm": "__summary__",
                "distillation_variant": "",
                "heldout_episode": "",
                "transfer_reason": "no_bad_episodes_identified",
            }
        ]
    episodes = _episode_labels(panel)
    ts = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce").reset_index(drop=True)
    y = np.asarray(y, dtype=np.int8)
    baseline_pred = np.asarray(baseline_pred, dtype=np.float32)
    returns = np.asarray(returns, dtype=np.float32)
    x_best = arms[best_arm]
    for episode_i, episode in enumerate(sorted(bad_episodes), start=1):
        holdout = episodes.eq(str(episode)).to_numpy(dtype=bool)
        test_idx = np.flatnonzero(holdout & (y >= 0))
        if len(test_idx) < 50:
            rows.append(
                {
                    "head": head,
                    "arm": "__summary__",
                    "distillation_variant": "",
                    "heldout_episode": str(episode),
                    "transfer_reason": "skipped_insufficient_holdout_rows",
                    "transfer_test_rows": int(len(test_idx)),
                    "training_target": "y_bin",
                    "forbidden_targets_used": False,
                }
            )
            continue
        train_mask = (y >= 0) & ~holdout
        if int(embargo_hours) > 0:
            hold_ts = ts.loc[holdout]
            if not hold_ts.empty:
                start = hold_ts.min() - pd.Timedelta(hours=int(embargo_hours))
                end = hold_ts.max() + pd.Timedelta(hours=int(embargo_hours))
                train_mask &= ~((ts >= start) & (ts <= end)).to_numpy(dtype=bool)
        train_idx = np.flatnonzero(train_mask)
        if len(train_idx) < 200:
            rows.append(
                {
                    "head": head,
                    "arm": "__summary__",
                    "distillation_variant": "",
                    "heldout_episode": str(episode),
                    "transfer_reason": "skipped_insufficient_train_rows",
                    "transfer_train_rows": int(len(train_idx)),
                    "transfer_test_rows": int(len(test_idx)),
                    "training_target": "y_bin",
                    "forbidden_targets_used": False,
                }
            )
            continue
        episode_pred: dict[str, np.ndarray] = {}
        episode_fit: dict[str, dict[str, Any]] = {}
        for spec_i, arm in enumerate(FEATURE_ARMS, start=1):
            if arm == ARM_A:
                pred = np.full(len(y), np.nan, dtype=np.float32)
                pred[test_idx] = baseline_pred[test_idx]
                variant = "unchanged_current_meta"
                fit_reason = "unchanged_current_meta_reference"
                train_rows = len(train_idx)
                feature_count = 1
            else:
                one_fold = [FoldContext(train_idx=train_idx, valid_idx=test_idx, fold_id=1)]
                pred, fit_rows = _fit_predict_classifier(
                    arms[arm],
                    y,
                    one_fold,
                    timestamps=panel["timestamp"],
                    seed=int(seed + episode_i * 1009 + spec_i * 917),
                    max_train_rows=int(max_train_rows),
                    max_depth=2 if arm == ARM_E else 3,
                )
                variant = "hard_label_context_arm"
                fit_info = fit_rows[0] if fit_rows else {}
                fit_reason = str(fit_info.get("reason", ""))
                train_rows = int(fit_info.get("train_rows", len(train_idx)) or 0)
                feature_count = int(fit_info.get("feature_count", 0) or 0)
            episode_pred[arm] = pred
            episode_fit[arm] = {
                "fit_reason": fit_reason,
                "train_rows": int(train_rows),
                "feature_count": int(feature_count),
            }
            metrics = _overall_metrics(
                head=head,
                arm=arm,
                variant=variant,
                y=y[test_idx],
                pred=pred[test_idx],
                baseline_pred=baseline_pred[test_idx],
                returns=returns[test_idx],
            )
            rows.append(
                {
                    "head": head,
                    "arm": arm,
                    "distillation_variant": variant,
                    "heldout_episode": str(episode),
                    "transfer_reason": fit_reason,
                    "transfer_train_rows": int(train_rows),
                    "transfer_test_rows": int(len(test_idx)),
                    "transfer_feature_count": int(feature_count),
                    "training_target": "y_bin",
                    "forbidden_targets_used": False,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key not in {"head", "arm", "distillation_variant"}
                    },
                }
            )
        for variant_i, variant in enumerate(DISTILLATION_VARIANTS, start=1):
            one_fold = [FoldContext(train_idx=train_idx, valid_idx=test_idx, fold_id=1)]
            if variant == "hard_label_only":
                pred = episode_pred[best_arm].copy()
                fit_info = episode_fit.get(best_arm, {})
                fit_reason = f"exact_reproduction_of_{best_arm}"
                train_rows = int(fit_info.get("train_rows", len(train_idx)) or 0)
                feature_count = int(fit_info.get("feature_count", 0) or 0)
            else:
                pred, fit_rows = _fit_predict_distilled(
                    x_best,
                    y,
                    baseline_pred,
                    canonical,
                    one_fold,
                    timestamps=panel["timestamp"],
                    seed=int(seed + episode_i * 1009 + (100 + variant_i) * 917),
                    max_train_rows=int(max_train_rows),
                    lambda0=float(distillation_lambda),
                    variant=variant,
                    max_depth=3,
                )
                fit_info = fit_rows[0] if fit_rows else {}
                fit_reason = str(fit_info.get("reason", ""))
                train_rows = int(fit_info.get("train_rows", len(train_idx)) or 0)
                feature_count = int(fit_info.get("feature_count", 0) or 0)
            dist_arm = f"F_{best_arm}__{variant}"
            metrics = _overall_metrics(
                head=head,
                arm=dist_arm,
                variant=variant,
                y=y[test_idx],
                pred=pred[test_idx],
                baseline_pred=baseline_pred[test_idx],
                returns=returns[test_idx],
            )
            rows.append(
                {
                    "head": head,
                    "arm": dist_arm,
                    "distillation_variant": variant,
                    "heldout_episode": str(episode),
                    "transfer_reason": fit_reason,
                    "transfer_train_rows": int(train_rows),
                    "transfer_test_rows": int(len(test_idx)),
                    "transfer_feature_count": int(feature_count),
                    "training_target": "y_bin",
                    "forbidden_targets_used": False,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key not in {"head", "arm", "distillation_variant"}
                    },
                }
            )
    return rows


def _oracle_specialist_leave_one_rows(
    *,
    head: str,
    panel: pd.DataFrame,
    arms: dict[str, pd.DataFrame | None],
    y: np.ndarray,
    baseline_pred: np.ndarray,
    returns: np.ndarray,
    bad_episodes: set[str],
    seed: int,
    max_train_rows: int,
    embargo_hours: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not bad_episodes or ARM_E not in arms or arms.get(ARM_E) is None:
        return rows
    episodes = _episode_labels(panel)
    ts = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce").reset_index(drop=True)
    y = np.asarray(y, dtype=np.int8)
    baseline_pred = np.asarray(baseline_pred, dtype=np.float32)
    returns = np.asarray(returns, dtype=np.float32)
    x_e = arms[ARM_E]
    for episode_i, episode in enumerate(sorted(bad_episodes), start=1):
        holdout = episodes.eq(str(episode)).to_numpy(dtype=bool)
        test_idx = np.flatnonzero(holdout & (y >= 0))
        if len(test_idx) < 50:
            rows.append(
                {
                    "head": head,
                    "heldout_episode": str(episode),
                    "benchmark_model": ARM_J,
                    "status": "skipped_insufficient_holdout_rows",
                    "test_rows": int(len(test_idx)),
                    "diagnostic_only": True,
                }
            )
            continue
        train_mask = (y >= 0) & ~holdout
        if int(embargo_hours) > 0:
            hold_ts = ts.loc[holdout]
            if not hold_ts.empty:
                start = hold_ts.min() - pd.Timedelta(hours=int(embargo_hours))
                end = hold_ts.max() + pd.Timedelta(hours=int(embargo_hours))
                train_mask &= ~((ts >= start) & (ts <= end)).to_numpy(dtype=bool)
        train_idx = np.flatnonzero(train_mask)
        one_fold = [FoldContext(train_idx=train_idx, valid_idx=test_idx, fold_id=1)]
        benchmark_preds: dict[str, tuple[np.ndarray, str, int]] = {}
        base = np.full(len(y), np.nan, dtype=np.float32)
        base[test_idx] = baseline_pred[test_idx]
        benchmark_preds[ARM_A] = (base, "unchanged_current_meta_reference", len(train_idx))
        pooled_pred, pooled_rows = _fit_predict_classifier(
            x_e,
            y,
            one_fold,
            timestamps=panel["timestamp"],
            seed=int(seed + episode_i * 1009 + 11),
            max_train_rows=int(max_train_rows),
            max_depth=2,
        )
        pooled_reason = str((pooled_rows[0] if pooled_rows else {}).get("reason", ""))
        benchmark_preds[ARM_E] = (pooled_pred, pooled_reason, int((pooled_rows[0] if pooled_rows else {}).get("train_rows", len(train_idx)) or 0))
        balance_weight = _regime_balance_weights(panel, bad_episodes)
        if balance_weight is not None:
            balanced_pred, balanced_rows = _fit_predict_classifier(
                x_e,
                y,
                one_fold,
                timestamps=panel["timestamp"],
                seed=int(seed + episode_i * 1009 + 23),
                max_train_rows=int(max_train_rows),
                max_depth=2,
                sample_weight=balance_weight,
            )
            balanced_info = balanced_rows[0] if balanced_rows else {}
            benchmark_preds[ARM_I] = (
                balanced_pred,
                str(balanced_info.get("reason", "")),
                int(balanced_info.get("train_rows", len(train_idx)) or 0),
            )
        bad_train_mask = train_mask & episodes.isin(set(str(x) for x in bad_episodes)).to_numpy(dtype=bool)
        specialist_train_idx = np.flatnonzero(bad_train_mask)
        if len(specialist_train_idx) >= 200 and len(np.unique(y[specialist_train_idx])) >= 2:
            specialist_fold = [FoldContext(train_idx=specialist_train_idx, valid_idx=test_idx, fold_id=1)]
            specialist_pred, specialist_rows = _fit_predict_classifier(
                x_e,
                y,
                specialist_fold,
                timestamps=panel["timestamp"],
                seed=int(seed + episode_i * 1009 + 37),
                max_train_rows=int(max_train_rows),
                max_depth=2,
            )
            specialist_info = specialist_rows[0] if specialist_rows else {}
            benchmark_preds[ARM_J] = (
                specialist_pred,
                str(specialist_info.get("reason", "oracle_bad_regime_specialist")),
                int(specialist_info.get("train_rows", len(specialist_train_idx)) or 0),
            )
        else:
            rows.append(
                {
                    "head": head,
                    "heldout_episode": str(episode),
                    "benchmark_model": ARM_J,
                    "status": "skipped_insufficient_bad_regime_training_rows",
                    "train_rows": int(len(specialist_train_idx)),
                    "test_rows": int(len(test_idx)),
                    "diagnostic_only": True,
                }
            )
        for model_name, (pred, reason, train_rows) in benchmark_preds.items():
            metrics = _overall_metrics(
                head=head,
                arm=model_name,
                variant="oracle_specialist_benchmark",
                y=y[test_idx],
                pred=pred[test_idx],
                baseline_pred=baseline_pred[test_idx],
                returns=returns[test_idx],
            )
            rows.append(
                {
                    "head": head,
                    "heldout_episode": str(episode),
                    "benchmark_model": model_name,
                    "status": "evaluated" if not reason else reason,
                    "train_rows": int(train_rows),
                    "test_rows": int(len(test_idx)),
                    "diagnostic_only": True,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if key not in {"head", "arm", "distillation_variant"}
                    },
                }
            )
    return rows


def _assemble_head_context(
    *,
    head: Any,
    panel: pd.DataFrame,
    race: Any,
    base_bundle: dict[str, Any],
    feature_dir: Path,
    transform_cache: Path | None,
    symbol_columns: dict[str, set[str]],
    regime_context: Path | None,
    max_regime_columns: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_x, _coverage, _summary = _assemble_selected_matrix(
        panel=panel,
        race=race,
        feature_dir=feature_dir,
        transform_cache=transform_cache,
        symbol_columns=symbol_columns,
    )
    base_selected_x = pd.DataFrame(index=panel.index)
    base_models, base_features = _base_models_for_head(base_bundle, head)
    if base_models and base_features:
        fake_race = type("FakeRace", (), {})()
        fake_best = type("FakeBest", (), {})()
        fake_best.selected_features = list(base_features)
        fake_best.get_training_meta_features = lambda: pd.DataFrame(index=panel.index)
        fake_best.model_effectiveness_history_defaults_ = {}
        fake_best.feature_stats_train = {}
        fake_race.best_model = fake_best
        base_selected_x, _base_cov, _base_summary = _assemble_selected_matrix(
            panel=panel,
            race=fake_race,
            feature_dir=feature_dir,
            transform_cache=transform_cache,
            symbol_columns=symbol_columns,
        )
    parts = [selected_x]
    if not base_selected_x.empty:
        extra = [c for c in base_selected_x.columns if c not in selected_x.columns]
        if extra:
            parts.append(base_selected_x[extra])
    export_x = _known_export_features(panel)
    regime_x = (
        _read_regime_features(regime_context, panel[["timestamp", "symbol"]], max_regime_columns)
        if regime_context is not None
        else pd.DataFrame(index=panel.index)
    )
    candidate_x = _merge_feature_candidates(pd.concat(parts, axis=1, copy=False), export_x, regime_x)
    raw = pd.concat([panel[["timestamp", "symbol"]].reset_index(drop=True), candidate_x.reset_index(drop=True)], axis=1)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    return _downcast_numeric(candidate_x), raw


def _head_priority_notes() -> pd.DataFrame:
    rows = [
        {
            "head": "long_dist",
            "priority": "prediction support; score-path instability; regime similarity; OI/funding crowding; liquidity; support x crowding",
        },
        {
            "head": "short_boll",
            "priority": "DAE state; relative-value dislocation; funding momentum; OI-to-volume; path instability; DAE/path x relative-value",
        },
        {
            "head": "short_asset",
            "priority": "prediction support; leverage/OI/funding state; market OI context; liquidity; support x crowding",
        },
        {
            "head": "long_bars",
            "priority": "tail-volatility; liquidity/volume; funding/OI residual pressure; prediction support",
        },
    ]
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> Path:
    forbidden = set(str(x).lower() for x in FORBIDDEN_TRAINING_TARGETS)
    if any(str(x).lower() in forbidden for x in (args.target_name,)):
        raise RuntimeError("Forbidden failure/payoff target requested")
    out_dir = _ensure_dir(Path(args.output_dir))
    meta_artifact_dir = Path(args.meta_artifact_dir)
    baseline_artifact_dir = Path(args.baseline_artifact_dir)
    feature_dir = Path(args.feature_dir)
    report_dir = Path(args.report_dir)
    transform_cache = Path(args.transform_cache) if args.transform_cache else None
    regime_context = Path(args.regime_context) if args.regime_context else None
    canonical_defs = _load_canonical_definitions(Path(args.canonical_reduction))
    if not canonical_defs:
        raise RuntimeError("No canonical definitions could be loaded")

    meta_state = joblib.load(meta_artifact_dir / "models" / "model_state_meta.pkl")
    meta_models = meta_state["bundle"]["meta_models"]
    heads = _discover_heads(meta_artifact_dir, report_dir, meta_models)
    wanted = set(HEADS)
    if args.only_head:
        wanted &= {str(x) for x in args.only_head}
    heads = [h for h in heads if h.head in wanted]
    with (baseline_artifact_dir / "base_models_intermediate.pkl").open("rb") as fh:
        base_bundle = pickle.load(fh)
    symbol_columns = _feature_store_union(feature_dir)

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    leave_one_rows: list[dict[str, Any]] = []
    gradient_conflict_rows: list[dict[str, Any]] = []
    oracle_specialist_rows: list[dict[str, Any]] = []
    bad_episode_context_rows: list[dict[str, Any]] = []
    directional_summary_rows: list[dict[str, Any]] = []
    directional_timestamp_rows: list[dict[str, Any]] = []
    directional_episode_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    fresh_start = pd.to_datetime(args.fresh_oos_start, utc=True, errors="coerce") if args.fresh_oos_start else None

    for head in heads:
        print(f"[one_head_context] processing head={head.head}", flush=True)
        panel = _normalise_keys(pd.read_parquet(head.meta_oof_path))
        panel = _downcast_numeric(panel, exclude=["timestamp", "symbol"])
        if int(args.max_rows) > 0 and len(panel) > int(args.max_rows):
            ts = pd.to_datetime(panel["timestamp"], utc=True, errors="coerce")
            keep = np.linspace(0, len(panel) - 1, int(args.max_rows)).round().astype(int)
            panel = panel.iloc[np.unique(keep)].sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
            print(f"[one_head_context] sampled head={head.head} rows={len(panel)} from chronological grid", flush=True)
        race = meta_models[head.meta_key]
        current_x, raw = _assemble_head_context(
            head=head,
            panel=panel,
            race=race,
            base_bundle=base_bundle,
            feature_dir=feature_dir,
            transform_cache=transform_cache,
            symbol_columns=symbol_columns,
            regime_context=regime_context,
            max_regime_columns=int(args.max_regime_columns),
        )
        contract = _candidate_feature_contract(current_x)
        disallowed_current = contract.loc[contract["allowed_by_clean_contract"].astype(bool).ne(True)]
        y = _meta_target(panel)
        baseline_pred = _current_meta_score(panel)
        returns = _pick_realized_return(panel).to_numpy(dtype=np.float32, copy=False)

        dev_idx = np.arange(len(panel), dtype=np.int64)
        fresh_split: dict[str, Any] | None = None
        if fresh_start is not None and pd.notna(fresh_start):
            fresh_split = _fresh_oos_indices(panel["timestamp"], pd.Timestamp(fresh_start), embargo_hours=int(args.embargo_hours))
            dev_idx = np.asarray(fresh_split["train_idx"], dtype=np.int64)
        if len(dev_idx) < 500:
            continue
        panel_dev = panel.iloc[dev_idx].reset_index(drop=True)
        raw_dev = raw.iloc[dev_idx].reset_index(drop=True)
        current_x_dev = current_x.iloc[dev_idx].reset_index(drop=True)
        y_dev = y[dev_idx]
        baseline_dev = baseline_pred[dev_idx]
        returns_dev = returns[dev_idx]
        folds = _make_chrono_folds(panel_dev["timestamp"], int(args.outer_folds), embargo_hours=int(args.embargo_hours))
        canonical, ctx_diag = _fold_canonical_features(
            raw_dev,
            folds,
            canonical_defs,
            trailing_window=int(args.trailing_window),
            min_periods=int(args.min_periods),
            min_resolved_features=int(args.min_resolved_features),
        )
        canonical["leaf_occupancy_novelty"] = _fit_leaf_occupancy_novelty(panel_dev, folds).reset_index(drop=True)
        canonical = canonical.loc[:, list(CANONICAL_CONTEXT)]
        canonical.to_parquet(out_dir / f"{head.head}_fold_fitted_one_head_context.parquet", index=False)
        for diag in ctx_diag:
            context_rows.append(
                {
                    "head": head.head,
                    "fold": diag["fold"],
                    "train_rows": diag["train_rows"],
                    "valid_rows": diag["valid_rows"],
                    "train_start": diag["train_start"],
                    "train_end": diag["train_end"],
                    "valid_start": diag["valid_start"],
                    "valid_end": diag["valid_end"],
                    "valid_output_feature_count": int(diag["valid_diagnostics"].get("output_feature_count", 0)) + 1,
                    "leaf_occupancy_novelty_fold_fitted": True,
                }
            )
        bad_episodes, bad_meta = _load_episode_registry(args.episode_registry, head=head.head, target_name=args.target_name)
        if not bad_episodes and bool(bad_meta.get("fallback_allowed", False)):
            weekly = _weekly_high_conf_metrics(panel_dev, float(args.rank_threshold), int(args.min_week_rows))
            bad_weeks, fallback_meta = _bad_recent_weeks(
                weekly, recent_weeks=int(args.recent_weeks), min_week_rows=int(args.min_week_rows)
            )
            bad_episodes = {pd.Timestamp(w).strftime("%Y-%m-%d") for w in bad_weeks}
            bad_meta = {
                **fallback_meta,
                "source": str(args.episode_registry),
                "reason": f"fallback_bad_recent_weeks_after_{bad_meta.get('reason', 'registry_unavailable')}",
                "fallback_allowed": True,
            }
        head_directional_timestamp_rows: list[dict[str, Any]] = []

        def record_directional(arm_name: str, variant_name: str, pred_values: np.ndarray) -> dict[str, Any]:
            ts_metrics = _directional_timestamp_metrics(
                head=head.head,
                arm=arm_name,
                variant=variant_name,
                panel=panel_dev,
                y=y_dev,
                pred=pred_values,
                baseline_pred=baseline_dev,
                returns=returns_dev,
                bad_episodes=bad_episodes,
                rank_threshold=float(args.rank_threshold),
                min_timestamp_rows=int(args.directional_min_timestamp_rows),
            )
            if ts_metrics.empty:
                return {}
            records = ts_metrics.to_dict(orient="records")
            directional_timestamp_rows.extend(records)
            head_directional_timestamp_rows.extend(records)
            agg = _directional_aggregate(ts_metrics)
            if agg.empty:
                return {}
            row = agg.iloc[0].to_dict()
            directional_summary_rows.append(row)
            return row

        arms = _arm_frames(panel_dev, current_x_dev, canonical)
        fold_valid_mask = np.zeros(len(panel_dev), dtype=bool)
        for fold in folds:
            fold_valid_mask[np.asarray(fold.valid_idx, dtype=np.int64)] = True
        baseline_fold_scored = baseline_dev.copy()
        baseline_fold_scored[~fold_valid_mask] = np.nan
        arm_predictions: dict[str, np.ndarray] = {ARM_A: baseline_fold_scored}
        arm_fit_rows: dict[str, list[dict[str, Any]]] = {
            ARM_A: [
                {
                    "fold": fold.fold_id,
                    "reason": "unchanged_current_meta_reference",
                    "feature_count": 1,
                    "training_target": args.target_name,
                }
                for fold in folds
            ]
        }
        for arm_i, arm in enumerate(FEATURE_ARMS, start=1):
            if arm == ARM_A:
                fold_rows.extend(
                    {
                        "head": head.head,
                        "arm": arm,
                        "distillation_variant": "unchanged_current_meta",
                        "fold": fold.fold_id,
                        "reason": "unchanged_current_meta_reference",
                        "feature_count": 1,
                        "training_target": args.target_name,
                    }
                    for fold in folds
                )
            else:
                pred, rows = _fit_predict_classifier(
                    arms[arm],
                    y_dev,
                    folds,
                    timestamps=panel_dev["timestamp"],
                    seed=int(args.seed + arm_i * 1009),
                    max_train_rows=int(args.max_train_rows),
                    max_depth=2 if arm == ARM_E else 3,
                )
                arm_predictions[arm] = pred
                arm_fit_rows[arm] = rows
                fold_rows.extend(
                    {
                        "head": head.head,
                        "arm": arm,
                        "distillation_variant": "hard_label_context_arm",
                        "training_target": args.target_name,
                        **row,
                    }
                    for row in rows
                )
            pred = arm_predictions[arm]
            summary = _overall_metrics(
                head=head.head,
                arm=arm,
                variant="unchanged_current_meta" if arm == ARM_A else "hard_label_context_arm",
                y=y_dev,
                pred=pred,
                baseline_pred=baseline_dev,
                returns=returns_dev,
            )
            summary.update(
                {
                    "training_target": args.target_name,
                    "single_output_score": True,
                    "forbidden_targets_used": False,
                    "canonical_model_state_features": len(MODEL_STATE),
                    "canonical_market_state_features": len(MARKET_STATE),
                    "disallowed_current_feature_count": int(len(disallowed_current)),
                    "fold_fitted_context": arm != ARM_A,
                    "fresh_oos_requested": bool(fresh_split is not None),
                    "fresh_oos_status": "not_evaluated" if fresh_split is None else "reserved",
                    "bad_episode_count": int(len(bad_episodes)),
                    "bad_episode_reason": bad_meta.get("reason", ""),
                    "episode_registry_source": bad_meta.get("source", ""),
                    "recommendation": "reference" if arm == ARM_A else "research_only",
                }
            )
            summary_rows.append(summary)
            directional_summary = record_directional(
                str(arm),
                str(summary["distillation_variant"]),
                pred,
            )
            if directional_summary:
                summary.update({f"directional_{k}": v for k, v in directional_summary.items() if k not in {"head", "arm", "distillation_variant"}})
            cell_rows.extend(
                _build_cell_rows(
                    head=head.head,
                    arm=arm,
                    variant=summary["distillation_variant"],
                    panel=panel_dev,
                    canonical=canonical,
                    y=y_dev,
                    pred=pred,
                    returns=returns_dev,
                    bad_episodes=bad_episodes,
                )
            )
        non_ref = [row for row in summary_rows if row["head"] == head.head and row["arm"] != ARM_A]
        feature_directional = {
            str(row.get("arm")): row
            for row in directional_summary_rows
            if row.get("head") == head.head and str(row.get("arm")) in set(FEATURE_ARMS)
        }
        best = max(
            non_ref,
            key=lambda row: _directional_selection_tuple({**row, **feature_directional.get(str(row.get("arm")), {})}),
        )
        best_arm = str(best["arm"])
        best_rows.append(
            {
                "head": head.head,
                "best_feature_arm": best_arm,
                "selection_metric": "lexicographic_directional_hr30_ndcg_then_logloss",
                **{
                    f"selected_{key}": value
                    for key, value in feature_directional.get(best_arm, {}).items()
                    if key
                    in {
                        "delta_timestamp_weighted_hr_top10",
                        "delta_timestamp_weighted_hr_top20",
                        "delta_timestamp_weighted_hr_top30",
                        "delta_ndcg_top30",
                        "delta_average_precision_top30",
                    }
                },
            }
        )
        x_best = arms[best_arm]
        for variant_i, variant in enumerate(DISTILLATION_VARIANTS, start=1):
            if variant == "hard_label_only":
                pred = arm_predictions[best_arm].copy()
                rows = [
                    {
                        **row,
                        "reason": f"exact_reproduction_of_{best_arm}",
                        "reproduced_arm": best_arm,
                    }
                    for row in arm_fit_rows.get(best_arm, [])
                ]
            else:
                pred, rows = _fit_predict_distilled(
                    x_best,
                    y_dev,
                    baseline_dev,
                    canonical,
                    folds,
                    timestamps=panel_dev["timestamp"],
                    seed=int(args.seed + 5000 + variant_i * 1009),
                    max_train_rows=int(args.max_train_rows),
                    lambda0=float(args.distillation_lambda),
                    variant=variant,
                    max_depth=3,
                )
            dist_arm = f"F_{best_arm}__{variant}"
            fold_rows.extend(
                {
                    "head": head.head,
                    "arm": dist_arm,
                    "distillation_variant": variant,
                    "training_target": args.target_name,
                    **row,
                }
                for row in rows
            )
            summary = _overall_metrics(
                head=head.head,
                arm=dist_arm,
                variant=variant,
                y=y_dev,
                pred=pred,
                baseline_pred=baseline_dev,
                returns=returns_dev,
            )
            summary.update(
                {
                    "training_target": args.target_name,
                    "single_output_score": True,
                    "forbidden_targets_used": False,
                    "best_feature_arm": best_arm,
                    "fold_fitted_context": True,
                    "fresh_oos_requested": bool(fresh_split is not None),
                    "fresh_oos_status": "not_evaluated" if fresh_split is None else "reserved",
                    "bad_episode_count": int(len(bad_episodes)),
                    "bad_episode_reason": bad_meta.get("reason", ""),
                    "episode_registry_source": bad_meta.get("source", ""),
                    "recommendation": "research_only",
                }
            )
            summary_rows.append(summary)
            record_directional(dist_arm, str(variant), pred)
            cell_rows.extend(
                _build_cell_rows(
                    head=head.head,
                    arm=dist_arm,
                    variant=variant,
                    panel=panel_dev,
                    canonical=canonical,
                    y=y_dev,
                    pred=pred,
                    returns=returns_dev,
                    bad_episodes=bad_episodes,
                )
            )
        for context_arm in CONTEXTUAL_SCORE_ARMS:
            if context_arm == ARM_G:
                pred, rows = _fit_predict_rank_preserving_calibration(
                    canonical,
                    y_dev,
                    baseline_dev,
                    folds,
                    timestamps=panel_dev["timestamp"],
                )
                variant = "rank_preserving_timestamp_logit_shift"
            else:
                pred, rows = _fit_predict_contextual_correction(
                    canonical,
                    y_dev,
                    baseline_dev,
                    folds,
                    timestamps=panel_dev["timestamp"],
                )
                variant = "timestamp_shift_plus_regularized_model_state_delta"
            arm_predictions[context_arm] = pred
            fold_rows.extend(
                {
                    "head": head.head,
                    "arm": context_arm,
                    "distillation_variant": variant,
                    "training_target": args.target_name,
                    **row,
                }
                for row in rows
            )
            summary = _overall_metrics(
                head=head.head,
                arm=context_arm,
                variant=variant,
                y=y_dev,
                pred=pred,
                baseline_pred=baseline_dev,
                returns=returns_dev,
            )
            summary.update(
                {
                    "training_target": args.target_name,
                    "single_output_score": True,
                    "forbidden_targets_used": False,
                    "fold_fitted_context": True,
                    "fresh_oos_requested": bool(fresh_split is not None),
                    "fresh_oos_status": "not_evaluated" if fresh_split is None else "reserved",
                    "bad_episode_count": int(len(bad_episodes)),
                    "bad_episode_reason": bad_meta.get("reason", ""),
                    "episode_registry_source": bad_meta.get("source", ""),
                    "recommendation": "research_only",
                }
            )
            summary_rows.append(summary)
            record_directional(context_arm, str(variant), pred)
            cell_rows.extend(
                _build_cell_rows(
                    head=head.head,
                    arm=context_arm,
                    variant=variant,
                    panel=panel_dev,
                    canonical=canonical,
                    y=y_dev,
                    pred=pred,
                    returns=returns_dev,
                    bad_episodes=bad_episodes,
                )
            )
        balance_weight = _regime_balance_weights(panel_dev, bad_episodes)
        if balance_weight is not None:
            pred, rows = _fit_predict_classifier(
                arms[ARM_E],
                y_dev,
                folds,
                timestamps=panel_dev["timestamp"],
                seed=int(args.seed + 8000),
                max_train_rows=int(args.max_train_rows),
                max_depth=2,
                sample_weight=balance_weight,
            )
            arm_predictions[ARM_I] = pred
            variant = "regime_balanced_hard_label_context_arm"
            fold_rows.extend(
                {
                    "head": head.head,
                    "arm": ARM_I,
                    "distillation_variant": variant,
                    "training_target": args.target_name,
                    **row,
                }
                for row in rows
            )
            summary = _overall_metrics(
                head=head.head,
                arm=ARM_I,
                variant=variant,
                y=y_dev,
                pred=pred,
                baseline_pred=baseline_dev,
                returns=returns_dev,
            )
            summary.update(
                {
                    "training_target": args.target_name,
                    "single_output_score": True,
                    "forbidden_targets_used": False,
                    "fold_fitted_context": True,
                    "fresh_oos_requested": bool(fresh_split is not None),
                    "fresh_oos_status": "not_evaluated" if fresh_split is None else "reserved",
                    "bad_episode_count": int(len(bad_episodes)),
                    "bad_episode_reason": bad_meta.get("reason", ""),
                    "episode_registry_source": bad_meta.get("source", ""),
                    "recommendation": "research_only_regime_balanced_diagnostic",
                }
            )
            summary_rows.append(summary)
            record_directional(ARM_I, str(variant), pred)
            cell_rows.extend(
                _build_cell_rows(
                    head=head.head,
                    arm=ARM_I,
                    variant=variant,
                    panel=panel_dev,
                    canonical=canonical,
                    y=y_dev,
                    pred=pred,
                    returns=returns_dev,
                    bad_episodes=bad_episodes,
                )
            )
        gradient_conflict = _gradient_conflict_diagnostics(
            head=head.head,
            panel=panel_dev,
            y=y_dev,
            predictions={
                arm: pred
                for arm, pred in arm_predictions.items()
                if arm in {ARM_A, ARM_B, ARM_C, ARM_D, ARM_E, ARM_G, ARM_H, ARM_I}
            },
            bad_episodes=bad_episodes,
        )
        if not gradient_conflict.empty:
            gradient_conflict_rows.extend(gradient_conflict.to_dict(orient="records"))
        episode_diag = _bad_episode_context_diagnostics(
            head=head.head,
            panel=panel_dev,
            canonical=canonical,
            y=y_dev,
            baseline_pred=baseline_dev,
            returns=returns_dev,
            predictions=arm_predictions,
            bad_episodes=bad_episodes,
            selected_arms=[arm for arm in [best_arm, ARM_D, ARM_E, ARM_I] if arm in arm_predictions],
        )
        if not episode_diag.empty:
            bad_episode_context_rows.extend(episode_diag.to_dict(orient="records"))
        if not bool(args.skip_leave_one):
            leave_one_rows.extend(
                _leave_one_bad_episode_rows(
                    head=head.head,
                    panel=panel_dev,
                    arms=arms,
                    best_arm=best_arm,
                    canonical=canonical,
                    y=y_dev,
                    baseline_pred=baseline_dev,
                    returns=returns_dev,
                    bad_episodes=bad_episodes,
                    seed=int(args.seed + 9000),
                    max_train_rows=int(args.max_train_rows),
                    embargo_hours=int(args.embargo_hours),
                    distillation_lambda=float(args.distillation_lambda),
                )
            )
        if not bool(args.skip_oracle_specialist):
            oracle_specialist_rows.extend(
                _oracle_specialist_leave_one_rows(
                    head=head.head,
                    panel=panel_dev,
                    arms=arms,
                    y=y_dev,
                    baseline_pred=baseline_dev,
                    returns=returns_dev,
                    bad_episodes=bad_episodes,
                    seed=int(args.seed + 12000),
                    max_train_rows=int(args.max_train_rows),
                    embargo_hours=int(args.embargo_hours),
                )
            )
        if head_directional_timestamp_rows:
            episode_directional = _directional_episode_metrics(pd.DataFrame(head_directional_timestamp_rows), bad_episodes)
            if not episode_directional.empty:
                directional_episode_rows.extend(episode_directional.to_dict(orient="records"))
        if fresh_split is not None:
            context_rows.append(
                {
                    "head": head.head,
                    "fold": "fresh_oos_reserved",
                    "train_rows": int(len(fresh_split["train_idx"])),
                    "valid_rows": int(len(fresh_split["test_idx"])),
                    "train_start": fresh_split.get("train_start", ""),
                    "train_end": fresh_split.get("train_end", ""),
                    "valid_start": fresh_split.get("test_start", ""),
                    "valid_end": fresh_split.get("test_end", ""),
                    "valid_output_feature_count": 0,
                    "leaf_occupancy_novelty_fold_fitted": True,
                    "note": "fresh OOS reserved until a fixed feature and distillation specification is selected",
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)
    cell_df = pd.DataFrame(cell_rows)
    context_df = pd.DataFrame(context_rows)
    leave_one_df = pd.DataFrame(leave_one_rows)
    gradient_conflict_df = pd.DataFrame(gradient_conflict_rows)
    oracle_specialist_df = pd.DataFrame(oracle_specialist_rows)
    bad_episode_context_df = pd.DataFrame(bad_episode_context_rows)
    directional_timestamp_df = pd.DataFrame(directional_timestamp_rows)
    directional_df = pd.DataFrame(directional_summary_rows)
    directional_episode_df = pd.DataFrame(directional_episode_rows)
    best_df = pd.DataFrame(best_rows)
    priority_df = _head_priority_notes()
    conflict_df = _period_conflict_diagnostics(cell_df)
    episode_ci_df = _episode_block_confidence_intervals(leave_one_df, seed=int(args.seed) + 15000)
    directional_episode_ci_df = _directional_episode_block_confidence_intervals(
        directional_episode_df,
        seed=int(args.seed) + 17500,
    )
    oracle_df = _oracle_period_specialist_benchmark(cell_df)
    promotion_df = _promotion_table(
        summary_df,
        conflict_df,
        episode_ci_df,
        gradient_conflict_df,
        oracle_specialist_df,
        directional_df,
        directional_episode_ci_df,
        winner_epsilon=float(args.winner_epsilon),
        lower_tail_tolerance=float(args.lower_tail_tolerance),
        directional_hr10_tolerance=float(args.directional_hr10_tolerance),
        directional_normal_tolerance=float(args.directional_normal_tolerance),
    )
    summary_df.to_csv(out_dir / "one_head_contextual_meta_ablation_summary.csv", index=False)
    fold_df.to_csv(out_dir / "one_head_contextual_meta_ablation_fold_metrics.csv", index=False)
    cell_df.to_csv(out_dir / "one_head_contextual_meta_ablation_cell_metrics.csv", index=False)
    context_df.to_csv(out_dir / "one_head_contextual_meta_ablation_context_diagnostics.csv", index=False)
    leave_one_df.to_csv(out_dir / "one_head_contextual_meta_ablation_leave_one_episode.csv", index=False)
    best_df.to_csv(out_dir / "one_head_contextual_meta_ablation_best_feature_arms.csv", index=False)
    priority_df.to_csv(out_dir / "one_head_contextual_meta_head_priorities.csv", index=False)
    conflict_df.to_csv(out_dir / "one_head_contextual_meta_period_conflict_diagnostics.csv", index=False)
    gradient_conflict_df.to_csv(out_dir / "one_head_contextual_meta_gradient_conflict_diagnostics.csv", index=False)
    episode_ci_df.to_csv(out_dir / "one_head_contextual_meta_episode_block_confidence_intervals.csv", index=False)
    directional_df.to_csv(out_dir / "one_head_contextual_meta_directional_metrics.csv", index=False)
    directional_timestamp_df.to_csv(out_dir / "one_head_contextual_meta_directional_timestamp_metrics.csv", index=False)
    directional_episode_df.to_csv(out_dir / "one_head_contextual_meta_directional_episode_metrics.csv", index=False)
    directional_episode_ci_df.to_csv(out_dir / "one_head_contextual_meta_directional_episode_confidence_intervals.csv", index=False)
    oracle_df.to_csv(out_dir / "one_head_contextual_meta_oracle_period_specialist.csv", index=False)
    oracle_specialist_df.to_csv(out_dir / "one_head_contextual_meta_oracle_specialist_leave_one.csv", index=False)
    bad_episode_context_df.to_csv(out_dir / "one_head_contextual_meta_bad_episode_context_diagnostics.csv", index=False)
    promotion_df.to_csv(out_dir / "one_head_contextual_meta_promotion_table.csv", index=False)
    audit = _requirement_audit(
        summary_df,
        fold_df,
        cell_df,
        context_df,
        leave_one_df,
        args,
        gradient_conflict=gradient_conflict_df,
        episode_ci=episode_ci_df,
        oracle_specialist=oracle_specialist_df,
        promotion=promotion_df,
        bad_episode_context=bad_episode_context_df,
        directional=directional_df,
        directional_timestamp=directional_timestamp_df,
        directional_episode=directional_episode_df,
        directional_episode_ci=directional_episode_ci_df,
    )
    (out_dir / "one_head_contextual_meta_ablation_requirement_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=_json_default)
    )
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=_json_default))
    _write_report(
        out_dir,
        summary_df,
        best_df,
        leave_one_df,
        audit,
        conflict_df,
        oracle_df,
        gradient_conflict_df,
        episode_ci_df,
        directional_df,
        directional_episode_df,
        directional_episode_ci_df,
        oracle_specialist_df,
        promotion_df,
        bad_episode_context_df,
    )
    print(f"[one_head_context] wrote results to {out_dir}", flush=True)
    return out_dir


def _requirement_audit(
    summary: pd.DataFrame,
    folds: pd.DataFrame,
    cells: pd.DataFrame,
    context: pd.DataFrame,
    leave_one: pd.DataFrame,
    args: argparse.Namespace,
    gradient_conflict: pd.DataFrame | None = None,
    episode_ci: pd.DataFrame | None = None,
    oracle_specialist: pd.DataFrame | None = None,
    promotion: pd.DataFrame | None = None,
    bad_episode_context: pd.DataFrame | None = None,
    directional: pd.DataFrame | None = None,
    directional_timestamp: pd.DataFrame | None = None,
    directional_episode: pd.DataFrame | None = None,
    directional_episode_ci: pd.DataFrame | None = None,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    heads = set(summary.get("head", pd.Series(dtype=str)).dropna().astype(str))
    arms = set(summary.get("arm", pd.Series(dtype=str)).dropna().astype(str))
    forbidden_used = bool(summary.get("forbidden_targets_used", pd.Series([True])).astype(bool).any())
    items.append(
        {
            "requirement": "same_label_single_output_contract",
            "status": "passed"
            if not forbidden_used and set(summary.get("training_target", pd.Series(dtype=str)).dropna()) == {"y_bin"}
            else "failed",
            "metrics": {
                "heads": len(heads),
                "training_targets": sorted(set(summary.get("training_target", pd.Series(dtype=str)).dropna())),
                "forbidden_targets_used": forbidden_used,
            },
        }
    )
    expected_feature_arms = set(FEATURE_ARMS)
    observed_feature_arms = {arm for arm in arms if str(arm) in expected_feature_arms}
    items.append(
        {
            "requirement": "feature_arms_A_to_E_present",
            "status": "passed" if observed_feature_arms == expected_feature_arms else "failed",
            "metrics": {
                "observed_feature_arms": sorted(observed_feature_arms),
                "expected_feature_arms": sorted(expected_feature_arms),
            },
        }
    )
    dist_variants = set(summary.get("distillation_variant", pd.Series(dtype=str)).dropna().astype(str)) & set(DISTILLATION_VARIANTS)
    items.append(
        {
            "requirement": "self_distillation_ablation_variants_present",
            "status": "passed" if dist_variants == set(DISTILLATION_VARIANTS) else "failed",
            "metrics": {"observed_variants": sorted(dist_variants), "expected_variants": sorted(DISTILLATION_VARIANTS)},
        }
    )
    observed_contextual_score_arms = {arm for arm in arms if str(arm) in set(CONTEXTUAL_SCORE_ARMS)}
    items.append(
        {
            "requirement": "contextual_score_arms_G_H_present",
            "status": "passed" if observed_contextual_score_arms == set(CONTEXTUAL_SCORE_ARMS) else "failed",
            "metrics": {
                "observed_arms": sorted(observed_contextual_score_arms),
                "expected_arms": sorted(CONTEXTUAL_SCORE_ARMS),
            },
        }
    )
    reproduction_rows: list[dict[str, Any]] = []
    group_iter = summary.groupby("head") if not summary.empty and "head" in summary.columns else []
    for head_name, group in group_iter:
        hard_rows = group.loc[group["distillation_variant"].astype(str).eq("hard_label_only")].copy()
        if hard_rows.empty:
            reproduction_rows.append({"head": head_name, "best_arm": "", "status": "missing_hard_label"})
            continue
        hard_row = hard_rows.iloc[0]
        hard_arm = str(hard_row["arm"])
        if not hard_arm.startswith("F_") or not hard_arm.endswith("__hard_label_only"):
            reproduction_rows.append({"head": head_name, "best_arm": "", "hard_arm": hard_arm, "status": "unparseable_hard_label_arm"})
            continue
        best_arm = hard_arm[len("F_") : -len("__hard_label_only")]
        best_match = group.loc[
            group["arm"].astype(str).eq(best_arm)
            & group["distillation_variant"].astype(str).isin({"hard_label_context_arm", "unchanged_current_meta"})
        ]
        if best_match.empty:
            reproduction_rows.append({"head": head_name, "best_arm": best_arm, "hard_arm": hard_arm, "status": "missing_selected_feature_arm"})
            continue
        best_row = best_match.iloc[0]
        reproduction_rows.append(
            {
                "head": head_name,
                "best_arm": best_arm,
                "hard_arm": hard_arm,
                "status": "checked",
                "rows_diff": abs(float(best_row.get("rows", np.nan)) - float(hard_row.get("rows", np.nan))),
                "log_loss_diff": abs(float(best_row.get("log_loss", np.nan)) - float(hard_row.get("log_loss", np.nan))),
                "brier_diff": abs(float(best_row.get("brier", np.nan)) - float(hard_row.get("brier", np.nan))),
                "auc_diff": abs(float(best_row.get("auc", np.nan)) - float(hard_row.get("auc", np.nan))),
            }
        )
    reproduction_ok = bool(reproduction_rows) and all(
        row.get("status") == "checked"
        and float(row.get("rows_diff", np.inf)) <= 0.0
        and float(row.get("log_loss_diff", np.inf)) <= 1e-12
        and float(row.get("brier_diff", np.inf)) <= 1e-12
        and float(row.get("auc_diff", np.inf)) <= 1e-12
        for row in reproduction_rows
    )
    items.append(
        {
            "requirement": "F_hard_label_exactly_reproduces_selected_feature_arm",
            "status": "passed" if reproduction_ok else "failed",
            "metrics": {"checks": reproduction_rows},
        }
    )
    cell_families = set(cells.get("cell_family", pd.Series(dtype=str)).dropna().astype(str))
    expected_cells = {
        "period_type",
        "support_quality_decile",
        "market_state_decile",
        "base_score_decile",
        "support_x_market_state_cell",
    }
    items.append(
        {
            "requirement": "conditional_cell_metrics_present",
            "status": "passed" if expected_cells <= cell_families else "failed",
            "metrics": {"cell_rows": int(len(cells)), "cell_families": sorted(cell_families)},
        }
    )
    context_ok = bool(
        not context.empty
        and pd.to_numeric(context.get("valid_output_feature_count", pd.Series([0])), errors="coerce").fillna(0).max()
        >= len(CANONICAL_CONTEXT)
    )
    items.append(
        {
            "requirement": "fold_fitted_canonical_context",
            "status": "passed" if context_ok else "failed",
            "metrics": {
                "context_rows": int(len(context)),
                "expected_context_features": len(CANONICAL_CONTEXT),
                "max_valid_output_feature_count": int(
                    pd.to_numeric(context.get("valid_output_feature_count", pd.Series([0])), errors="coerce")
                    .fillna(0)
                    .max()
                ),
            },
        }
    )
    fold_groups = folds.groupby(["head", "arm", "distillation_variant"])["fold"].nunique() if not folds.empty else pd.Series(dtype=float)
    items.append(
        {
            "requirement": "chronological_oof_fold_metrics_present",
            "status": "passed" if not fold_groups.empty and int(fold_groups.max()) >= int(args.outer_folds) else "failed",
            "metrics": {
                "fold_groups": int(len(fold_groups)),
                "outer_folds": int(args.outer_folds),
                "max_folds_per_group": int(fold_groups.max()) if not fold_groups.empty else 0,
                "embargo_hours": int(args.embargo_hours),
            },
        }
    )
    leave_one_evaluable = leave_one.loc[
        leave_one.get("arm", pd.Series(dtype=str)).astype(str).ne("__summary__")
    ] if not leave_one.empty else pd.DataFrame()
    registry_sources = sorted(set(summary.get("episode_registry_source", pd.Series(dtype=str)).dropna().astype(str)))
    eligible_episode_count = int(
        pd.to_numeric(summary.get("bad_episode_count", pd.Series([0])), errors="coerce").fillna(0).max()
    )
    items.append(
        {
            "requirement": "leave_one_bad_episode_evaluation_present",
            "status": "passed"
            if bool(getattr(args, "skip_leave_one", False)) or eligible_episode_count == 0 or not leave_one_evaluable.empty
            else "failed",
            "metrics": {
                "skipped": bool(getattr(args, "skip_leave_one", False)),
                "eligible_episode_count": eligible_episode_count,
                "rows": int(len(leave_one)),
                "evaluable_rows": int(len(leave_one_evaluable)),
                "heldout_episodes": int(leave_one_evaluable.get("heldout_episode", pd.Series(dtype=str)).nunique())
                if not leave_one_evaluable.empty
                else 0,
                "episode_registry_sources": registry_sources,
            },
        }
    )
    gradient_conflict = gradient_conflict if gradient_conflict is not None else pd.DataFrame()
    required_gradient_cols = {
        "leaf",
        "regime",
        "support",
        "gradient_sum",
        "hessian_sum",
        "optimal_update",
        "update_sign",
        "cancellation_score",
    }
    items.append(
        {
            "requirement": "score_region_gradient_conflict_diagnostics_present",
            "status": "passed" if not gradient_conflict.empty and required_gradient_cols <= set(gradient_conflict.columns) else "failed",
            "metrics": {
                "rows": int(len(gradient_conflict)),
                "required_columns_present": sorted(required_gradient_cols & set(gradient_conflict.columns)),
                "max_cancellation_score": float(
                    pd.to_numeric(gradient_conflict.get("cancellation_score", pd.Series(dtype=float)), errors="coerce")
                    .dropna()
                    .max()
                )
                if not gradient_conflict.empty
                else np.nan,
            },
        }
    )
    episode_ci = episode_ci if episode_ci is not None else pd.DataFrame()
    required_ci_cols = {"metric", "episode_count", "mean", "median", "ci05", "ci95", "ci_method"}
    items.append(
        {
            "requirement": "episode_block_confidence_intervals_present",
            "status": "passed"
            if bool(getattr(args, "skip_leave_one", False))
            or eligible_episode_count == 0
            or (not episode_ci.empty and required_ci_cols <= set(episode_ci.columns))
            else "failed",
            "metrics": {
                "rows": int(len(episode_ci)),
                "metrics": sorted(set(episode_ci.get("metric", pd.Series(dtype=str)).astype(str))) if not episode_ci.empty else [],
            },
        }
    )
    oracle_specialist = oracle_specialist if oracle_specialist is not None else pd.DataFrame()
    required_oracle_cols = {"heldout_episode", "benchmark_model", "status", "diagnostic_only"}
    evaluated_oracle = (
        oracle_specialist.get("status", pd.Series(dtype=str)).astype(str).eq("evaluated").sum()
        if not oracle_specialist.empty
        else 0
    )
    items.append(
        {
            "requirement": "oracle_specialist_benchmark_present",
            "status": "passed"
            if bool(getattr(args, "skip_oracle_specialist", False))
            or eligible_episode_count == 0
            or (not oracle_specialist.empty and required_oracle_cols <= set(oracle_specialist.columns))
            else "failed",
            "metrics": {
                "skipped": bool(getattr(args, "skip_oracle_specialist", False)),
                "rows": int(len(oracle_specialist)),
                "evaluated_rows": int(evaluated_oracle),
                "benchmark_models": sorted(set(oracle_specialist.get("benchmark_model", pd.Series(dtype=str)).astype(str)))
                if not oracle_specialist.empty
                else [],
            },
        }
    )
    promotion = promotion if promotion is not None else pd.DataFrame()
    directional = directional if directional is not None else pd.DataFrame()
    directional_timestamp = directional_timestamp if directional_timestamp is not None else pd.DataFrame()
    directional_episode = directional_episode if directional_episode is not None else pd.DataFrame()
    directional_episode_ci = directional_episode_ci if directional_episode_ci is not None else pd.DataFrame()
    required_directional_cols = {
        "head",
        "arm",
        "distillation_variant",
        "timestamp_weighted_hr_top10",
        "timestamp_weighted_hr_top20",
        "timestamp_weighted_hr_top30",
        "trade_weighted_hr_top30",
        "delta_timestamp_weighted_hr_top30",
        "ndcg_top30",
        "delta_ndcg_top30",
        "average_precision_top30",
        "top30_jaccard",
        "net_correct_trades_gained",
    }
    required_directional_timestamp_cols = {
        "head",
        "arm",
        "distillation_variant",
        "timestamp",
        "eligible_rows",
        "hr_top30",
        "baseline_hr_top30",
        "delta_hr_top30",
        "ndcg_top30",
        "top30_jaccard",
    }
    required_directional_episode_cols = {
        "head",
        "arm",
        "distillation_variant",
        "heldout_episode",
        "period_type",
        "delta_timestamp_weighted_hr_top30",
    }
    items.append(
        {
            "requirement": "directional_timestamp_local_top30_metrics_present",
            "status": "passed"
            if not directional.empty
            and required_directional_cols <= set(directional.columns)
            and not directional_timestamp.empty
            and required_directional_timestamp_cols <= set(directional_timestamp.columns)
            and (eligible_episode_count == 0 or (not directional_episode.empty and required_directional_episode_cols <= set(directional_episode.columns)))
            else "failed",
            "metrics": {
                "directional_rows": int(len(directional)),
                "timestamp_rows": int(len(directional_timestamp)),
                "episode_rows": int(len(directional_episode)),
                "episode_ci_rows": int(len(directional_episode_ci)),
                "required_directional_columns_present": sorted(required_directional_cols & set(directional.columns)),
                "required_timestamp_columns_present": sorted(required_directional_timestamp_cols & set(directional_timestamp.columns)),
                "required_episode_columns_present": sorted(required_directional_episode_cols & set(directional_episode.columns)),
            },
        }
    )
    required_promotion_cols = {
        "head",
        "arm",
        "delta_log_loss_improvement",
        "delta_brier_improvement",
        "timestamp_weighted_hr_top30",
        "delta_timestamp_weighted_hr_top30",
        "ndcg_top30",
        "delta_ndcg_top30",
        "passes_directional_pooled_constraints",
        "passes_directional_episode_constraints",
        "directional_promotion_candidate",
        "normal_period_delta_log_loss_improvement",
        "bad_period_delta_log_loss_improvement",
        "top10_delta_mean_return",
        "top10_delta_winner_magnitude",
        "top10_delta_lower_tail_return",
        "gradient_conflict_weighted",
        "gradient_conflict_high_row_fraction",
        "promotion_candidate",
    }
    items.append(
        {
            "requirement": "common_promotion_table_present",
            "status": "passed" if not promotion.empty and required_promotion_cols <= set(promotion.columns) else "failed",
            "metrics": {
                "rows": int(len(promotion)),
                "candidate_rows": int(
                    promotion.get("promotion_candidate", pd.Series(dtype=bool)).astype(bool).sum()
                )
                if not promotion.empty
                else 0,
                "required_columns_present": sorted(required_promotion_cols & set(promotion.columns)),
            },
        }
    )
    bad_episode_context = bad_episode_context if bad_episode_context is not None else pd.DataFrame()
    required_context_cols = {"head", "heldout_episode", "arm", "classification_hint", "context_missing_fraction"}
    items.append(
        {
            "requirement": "bad_episode_context_diagnostics_present",
            "status": "passed"
            if eligible_episode_count == 0 or (not bad_episode_context.empty and required_context_cols <= set(bad_episode_context.columns))
            else "failed",
            "metrics": {
                "eligible_episode_count": eligible_episode_count,
                "rows": int(len(bad_episode_context)),
                "classification_hints": sorted(
                    set(bad_episode_context.get("classification_hint", pd.Series(dtype=str)).astype(str))
                )
                if not bad_episode_context.empty
                else [],
            },
        }
    )
    statuses = [item["status"] for item in items]
    return {
        "status": "passed" if all(status == "passed" for status in statuses) else "failed",
        "items": items,
        "outcomes": {
            "summary_rows": int(len(summary)),
            "cell_rows": int(len(cells)),
            "leave_one_rows": int(len(leave_one)),
            "heads": sorted(heads),
            "best_by_head": summary.sort_values("delta_log_loss_improvement", ascending=False)
            .groupby("head", as_index=False)
            .head(1)[["head", "arm", "distillation_variant", "delta_log_loss_improvement", "delta_brier_improvement"]]
            .to_dict(orient="records")
            if not summary.empty and "delta_log_loss_improvement" in summary
            else [],
        },
    }


def _period_conflict_diagnostics(cells: pd.DataFrame) -> pd.DataFrame:
    if cells.empty or "cell_family" not in cells.columns:
        return pd.DataFrame()
    period = cells.loc[cells["cell_family"].astype(str).eq("period_type")].copy()
    if period.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for head, group in period.groupby("head", sort=True):
        baseline = group.loc[group["arm"].astype(str).eq(ARM_A)]
        if baseline.empty:
            continue
        base_by_cell = baseline.set_index("cell")
        for (arm, variant), arm_group in group.loc[~group["arm"].astype(str).eq(ARM_A)].groupby(
            ["arm", "distillation_variant"], sort=True
        ):
            arm_by_cell = arm_group.set_index("cell")
            bad_delta = np.nan
            normal_delta = np.nan
            if "bad_period" in base_by_cell.index and "bad_period" in arm_by_cell.index:
                bad_delta = float(base_by_cell.loc["bad_period", "log_loss"] - arm_by_cell.loc["bad_period", "log_loss"])
            if "normal_period" in base_by_cell.index and "normal_period" in arm_by_cell.index:
                normal_delta = float(
                    base_by_cell.loc["normal_period", "log_loss"] - arm_by_cell.loc["normal_period", "log_loss"]
                )
            rows.append(
                {
                    "head": head,
                    "arm": arm,
                    "distillation_variant": variant,
                    "bad_period_delta_log_loss_improvement": bad_delta,
                    "normal_period_delta_log_loss_improvement": normal_delta,
                    "period_objective_conflict": bool(
                        np.isfinite(bad_delta) and np.isfinite(normal_delta) and np.sign(bad_delta) != np.sign(normal_delta)
                    ),
                    "diagnostic_type": "gradient_conflict_proxy_from_period_logloss",
                }
            )
    return pd.DataFrame(rows)


def _oracle_period_specialist_benchmark(cells: pd.DataFrame) -> pd.DataFrame:
    if cells.empty or "cell_family" not in cells.columns:
        return pd.DataFrame()
    period = cells.loc[cells["cell_family"].astype(str).eq("period_type")].copy()
    if period.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for head, group in period.groupby("head", sort=True):
        baseline = group.loc[group["arm"].astype(str).eq(ARM_A)]
        if baseline.empty:
            continue
        base_by_cell = baseline.set_index("cell")
        base_total = 0.0
        total_count = 0.0
        for cell, base_row in base_by_cell.iterrows():
            count = float(base_row.get("trade_count", 0.0))
            ll = float(base_row.get("log_loss", np.nan))
            if np.isfinite(ll) and count > 0:
                base_total += count * ll
                total_count += count
        if total_count <= 0:
            continue
        baseline_weighted = base_total / total_count
        for (arm, variant), arm_group in group.loc[~group["arm"].astype(str).eq(ARM_A)].groupby(
            ["arm", "distillation_variant"], sort=True
        ):
            arm_by_cell = arm_group.set_index("cell")
            oracle_total = 0.0
            oracle_count = 0.0
            choices: list[str] = []
            for cell, base_row in base_by_cell.iterrows():
                if cell not in arm_by_cell.index:
                    continue
                count = float(base_row.get("trade_count", 0.0))
                base_ll = float(base_row.get("log_loss", np.nan))
                arm_ll = float(arm_by_cell.loc[cell].get("log_loss", np.nan))
                if not np.isfinite(base_ll) or not np.isfinite(arm_ll) or count <= 0:
                    continue
                use_arm = arm_ll < base_ll
                oracle_total += count * (arm_ll if use_arm else base_ll)
                oracle_count += count
                choices.append(f"{cell}:{'arm' if use_arm else 'baseline'}")
            if oracle_count <= 0:
                continue
            oracle_weighted = oracle_total / oracle_count
            rows.append(
                {
                    "head": head,
                    "arm": arm,
                    "distillation_variant": variant,
                    "baseline_period_weighted_log_loss": baseline_weighted,
                    "oracle_period_weighted_log_loss": oracle_weighted,
                    "oracle_delta_log_loss_improvement": baseline_weighted - oracle_weighted,
                    "period_choices": ";".join(choices),
                    "diagnostic_only": True,
                }
            )
    return pd.DataFrame(rows)


def _write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    best: pd.DataFrame,
    leave_one: pd.DataFrame,
    audit: dict[str, Any],
    conflict: pd.DataFrame,
    oracle: pd.DataFrame,
    gradient_conflict: pd.DataFrame,
    episode_ci: pd.DataFrame,
    directional: pd.DataFrame,
    directional_episode: pd.DataFrame,
    directional_episode_ci: pd.DataFrame,
    oracle_specialist: pd.DataFrame,
    promotion: pd.DataFrame,
    bad_episode_context: pd.DataFrame,
) -> None:
    lines = [
        "# One-Head Contextual Meta Ablation",
        "",
        "Diagnostic retraining matrix. No production model is modified.",
        "",
        "## Contract",
        "",
        "- Training target: unchanged `y_bin` only.",
        "- Output: one meta probability score per row.",
        "- Forbidden as training targets: `high_conf_miss`, `high_conf_tail_loss`, reliability heads, expected-return heads, quantile/payoff heads.",
        "- Archetypes/failure classifiers/leaf analyses are used only to choose context inputs.",
        "",
        "## Arms",
        "",
        f"- A: `{ARM_A}`",
        f"- B: `{ARM_B}`",
        f"- C: `{ARM_C}`",
        f"- D: `{ARM_D}`",
        f"- E: `{ARM_E}`",
        "- F: best feature arm plus self-distillation ablations.",
        f"- G: `{ARM_G}`; timestamp-level rank-preserving contextual calibration.",
        f"- H: `{ARM_H}`; G plus a heavily regularized model-state row correction.",
        f"- I: `{ARM_I}`; regime-balanced version of E, research-only.",
        "",
    ]
    lines.append("## Requirement Audit")
    lines.append("")
    lines.append(pd.DataFrame(audit.get("items", [])).to_markdown(index=False))
    lines.append("")
    if not best.empty:
        lines.append("## Best Feature Arms")
        lines.append("")
        lines.append(best.to_markdown(index=False))
        lines.append("")
    if not directional.empty:
        view_cols = [
            "head",
            "arm",
            "distillation_variant",
            "directional_timestamp_count",
            "directional_eligibility_source",
            "timestamp_weighted_hr_top10",
            "timestamp_weighted_hr_top20",
            "timestamp_weighted_hr_top30",
            "delta_timestamp_weighted_hr_top30",
            "trade_weighted_hr_top30",
            "ndcg_top30",
            "delta_ndcg_top30",
            "average_precision_top30",
            "delta_average_precision_top30",
            "worst_week_hr_top30",
            "q10_week_hr_top30",
            "top30_jaccard",
            "net_correct_trades_gained",
        ]
        lines.append("## Directional Ranking Metrics")
        lines.append("")
        lines.append(directional[[c for c in view_cols if c in directional.columns]].to_markdown(index=False, floatfmt=".5f"))
        lines.append("")
    if not directional_episode.empty:
        view_cols = [
            "head",
            "heldout_episode",
            "period_type",
            "arm",
            "timestamp_weighted_hr_top30",
            "delta_timestamp_weighted_hr_top30",
            "ndcg_top30",
            "delta_ndcg_top30",
            "worst_week_hr_top30",
            "q10_week_hr_top30",
            "net_correct_trades_gained",
        ]
        lines.append("## Directional Episode Metrics")
        lines.append("")
        lines.append(
            directional_episode[[c for c in view_cols if c in directional_episode.columns]]
            .sort_values(["head", "period_type", "heldout_episode", "arm"])
            .to_markdown(index=False, floatfmt=".5f")
        )
        lines.append("")
    if not directional_episode_ci.empty:
        view_cols = [
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
        ]
        lines.append("## Directional Episode-Block Confidence Intervals")
        lines.append("")
        lines.append(directional_episode_ci[[c for c in view_cols if c in directional_episode_ci.columns]].to_markdown(index=False, floatfmt=".5f"))
        lines.append("")
    if not summary.empty:
        show = [
            "head",
            "arm",
            "distillation_variant",
            "rows",
            "auc",
            "baseline_auc",
            "log_loss",
            "baseline_log_loss",
            "delta_log_loss_improvement",
            "brier",
            "delta_brier_improvement",
            "top10_delta_mean_return",
            "top10_delta_winner_magnitude",
            "top10_delta_lower_tail_return",
            "recommendation",
        ]
        lines.append("## Summary")
        lines.append("")
        lines.append(summary[[c for c in show if c in summary.columns]].to_markdown(index=False, floatfmt=".5f"))
        lines.append("")
    if not promotion.empty:
        view_cols = [
            "head",
            "arm",
            "timestamp_weighted_hr_top30",
            "delta_timestamp_weighted_hr_top30",
            "directional_episode_median_delta_timestamp_weighted_hr_top30",
            "directional_episode_positive_rate_delta_timestamp_weighted_hr_top30",
            "ndcg_top30",
            "delta_ndcg_top30",
            "timestamp_weighted_hr_top10",
            "delta_timestamp_weighted_hr_top10",
            "normal_period_delta_hr_top30",
            "bad_period_delta_hr_top30",
            "top30_delta_log_loss_on_selected",
            "top10_delta_mean_return",
            "top10_delta_winner_magnitude",
            "top10_delta_lower_tail_return",
            "passes_directional_pooled_constraints",
            "passes_directional_episode_constraints",
            "gradient_conflict_weighted",
            "gradient_conflict_high_row_fraction",
            "oracle_specialist_gap_log_loss",
            "promotion_candidate",
        ]
        lines.append("## Directional Promotion Table")
        lines.append("")
        show = promotion[[c for c in view_cols if c in promotion.columns]].copy()
        lines.append(show.to_markdown(index=False, floatfmt=".5f"))
        lines.append("")
    if not leave_one.empty:
        view_cols = [
            "head",
            "heldout_episode",
            "arm",
            "distillation_variant",
            "transfer_reason",
            "rows",
            "auc",
            "baseline_auc",
            "delta_log_loss_improvement",
            "delta_brier_improvement",
            "top10_delta_mean_return",
            "top10_delta_winner_magnitude",
            "top10_delta_lower_tail_return",
        ]
        lines.append("## Leave-One Bad Episode")
        lines.append("")
        lines.append(leave_one[[c for c in view_cols if c in leave_one.columns]].to_markdown(index=False, floatfmt=".5f"))
        lines.append("")
    if not conflict.empty:
        view_cols = [
            "head",
            "arm",
            "distillation_variant",
            "bad_period_delta_log_loss_improvement",
            "normal_period_delta_log_loss_improvement",
            "period_objective_conflict",
        ]
        lines.append("## Period Conflict Diagnostic")
        lines.append("")
        lines.append(conflict[[c for c in view_cols if c in conflict.columns]].to_markdown(index=False, floatfmt=".5f"))
        lines.append("")
    if not gradient_conflict.empty:
        view_cols = [
            "head",
            "arm",
            "leaf",
            "regime",
            "support",
            "gradient_sum",
            "hessian_sum",
            "optimal_update",
            "update_sign",
            "cancellation_score",
        ]
        lines.append("## Gradient Conflict Diagnostic")
        lines.append("")
        show = gradient_conflict.sort_values("cancellation_score", ascending=False).head(40)
        lines.append(show[[c for c in view_cols if c in show.columns]].to_markdown(index=False, floatfmt=".5f"))
        lines.append("")
    if not episode_ci.empty:
        view_cols = [
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
        ]
        lines.append("## Episode-Block Confidence Intervals")
        lines.append("")
        show = episode_ci.loc[episode_ci["metric"].astype(str).isin(["delta_log_loss_improvement", "delta_brier_improvement", "top10_delta_mean_return"])]
        lines.append(show[[c for c in view_cols if c in show.columns]].head(80).to_markdown(index=False, floatfmt=".5f"))
        lines.append("")
    if not oracle.empty:
        view_cols = [
            "head",
            "arm",
            "distillation_variant",
            "baseline_period_weighted_log_loss",
            "oracle_period_weighted_log_loss",
            "oracle_delta_log_loss_improvement",
            "period_choices",
        ]
        lines.append("## Oracle Period Specialist Benchmark")
        lines.append("")
        lines.append(oracle[[c for c in view_cols if c in oracle.columns]].to_markdown(index=False, floatfmt=".5f"))
        lines.append("")
    if not oracle_specialist.empty:
        view_cols = [
            "head",
            "heldout_episode",
            "benchmark_model",
            "status",
            "train_rows",
            "test_rows",
            "delta_log_loss_improvement",
            "delta_brier_improvement",
            "top10_delta_mean_return",
        ]
        lines.append("## Oracle Specialist Leave-One Benchmark")
        lines.append("")
        lines.append(oracle_specialist[[c for c in view_cols if c in oracle_specialist.columns]].to_markdown(index=False, floatfmt=".5f"))
        lines.append("")
    if not bad_episode_context.empty:
        view_cols = [
            "head",
            "heldout_episode",
            "arm",
            "rows",
            "label_rate",
            "delta_log_loss_improvement",
            "delta_brier_improvement",
            "top10_delta_mean_return",
            "top10_delta_lower_tail_return",
            "market_context_mean",
            "model_context_mean",
            "context_missing_fraction",
            "classification_hint",
        ]
        lines.append("## Bad Episode Context Diagnostics")
        lines.append("")
        lines.append(
            bad_episode_context[[c for c in view_cols if c in bad_episode_context.columns]]
            .sort_values(["head", "heldout_episode", "arm"])
            .to_markdown(index=False, floatfmt=".5f")
        )
        lines.append("")
    (out_dir / "one_head_contextual_meta_ablation_report.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-artifact-dir", default="data_perp/artifacts/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--baseline-artifact-dir", default="data_perp/artifacts/20260617_090000_no_mkt4_labelhpo_final_fit")
    parser.add_argument("--report-dir", default="data_perp/reports/meta_featureselect_recentguard_20260622_0119")
    parser.add_argument("--feature-dir", default="data_perp/features/20260605_070000")
    parser.add_argument(
        "--transform-cache",
        default="data_perp/reports/performance_regime_break_transform_cache/generated_transforms_single_3f7f9c53eaaa98ce632760a976691f24.parquet",
    )
    parser.add_argument("--canonical-reduction", default="data_perp/reports/meta_recent_failure_diagnostics_20260622_archetype_usefulness_multitarget_clean_contract_v1/canonical_archetype_reduction.csv")
    parser.add_argument("--regime-context", default="")
    parser.add_argument("--output-dir", default="data_perp/reports/one_head_contextual_meta_ablation_20260622")
    parser.add_argument("--target-name", default="y_bin")
    parser.add_argument("--rank-threshold", type=float, default=0.70)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--embargo-hours", type=int, default=24)
    parser.add_argument("--recent-weeks", type=int, default=8)
    parser.add_argument("--min-week-rows", type=int, default=30)
    parser.add_argument("--trailing-window", type=int, default=24 * 28)
    parser.add_argument("--min-periods", type=int, default=24 * 7)
    parser.add_argument("--min-resolved-features", type=int, default=2)
    parser.add_argument("--max-regime-columns", type=int, default=80)
    parser.add_argument("--max-train-rows", type=int, default=60000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--distillation-lambda", type=float, default=1.0)
    parser.add_argument("--winner-epsilon", type=float, default=0.0005)
    parser.add_argument("--lower-tail-tolerance", type=float, default=0.0010)
    parser.add_argument("--directional-min-timestamp-rows", type=int, default=5)
    parser.add_argument("--directional-hr10-tolerance", type=float, default=0.001)
    parser.add_argument("--directional-normal-tolerance", type=float, default=0.001)
    parser.add_argument("--fresh-oos-start", default="")
    parser.add_argument("--episode-registry", default=str(DEFAULT_EPISODE_REGISTRY))
    parser.add_argument("--skip-leave-one", action="store_true")
    parser.add_argument("--skip-oracle-specialist", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--only-head", nargs="*", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
