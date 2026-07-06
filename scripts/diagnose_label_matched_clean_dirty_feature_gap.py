#!/usr/bin/env python3
"""Matched clean-oracle vs dirty-false-positive proxy feature diagnostic.

This is a no-training diagnostic. It explains where the causal proxy selector
confuses clean oracle rows with dirty high-scoring false positives, using
within-bucket ranks so broad time/regime effects do not dominate the contrast.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.report_label_proxy_quality_with_economic_limits import (  # noqa: E402
    DEFAULT_FEATURE_LIST_CSV,
    DEFAULT_PROXY_TOP_K,
    _score_proxy,
)
from scripts.run_label_adverse_path_proxy_gate_ablation import (  # noqa: E402
    _adverse_path_composite_features,
    _path_targets,
)
from scripts.run_label_economic_proxy_ablation import _label_targets  # noqa: E402
from scripts.run_label_quality_proxy_diagnostics import (  # noqa: E402
    DEFAULT_FEATURE_DIR,
    _feature_columns,
    _json_safe,
    _load_feature_store_columns,
    _load_labels,
    _make_targets,
    _path_metrics,
    _rank_top_indices,
    _read_feature_list,
    _safe_mean,
    _safe_quantile,
    _selection_metrics,
    _spearman,
)
from scripts.run_soft_label_economic_proxy_ablation import (  # noqa: E402
    DEFAULT_EVENT_FEATURE_STORE_FEATURES,
    DEFAULT_PRIOR_WINDOWS_DAYS,
    DEFAULT_STATE_PATH_PRIOR_FEATURES,
    _causal_outcome_prior_features,
    _causal_state_path_prior_features,
    _event_confirmation_features,
)
from scripts.run_soft_label_candidate_source_ablation import (  # noqa: E402
    _build_sources,
    _source_context,
)

DEFAULT_LABELS_PATH = Path(
    "data_perp/artifacts/"
    "20260702_120500_first_touch_c0_fast6_s10_policy_net_labels_exitaligned/labels"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/label_matched_clean_dirty_feature_gap_v1")
DEFAULT_MONTHS = ("2026-04", "2026-05", "2026-06")
DEFAULT_LABEL_ARMS = (
    "S61_tpnet_strict_adverse_veto_rank",
    "S62_tpnet_clean_dirty_contrast_rank",
)
DEFAULT_STRICT_ROUNDA_LABEL_ARMS = (
    "S120_s3_clean_utility_veto",
    "S121_s8_clean_rank_veto",
    "S122_clean_dirty_contrast_rank",
    "S123_fast_clean_path_rank",
    "S124_s3_net_floor_veto",
    "S125_s8_net_floor_rank_veto",
    "S126_clean_net_direct_rank",
    "S127_fast_clean_net_rank",
)
DEFAULT_TOP_FRACS = (0.01, 0.03)
DEFAULT_MATCH_MODES = ("timestamp_side", "day_side", "regime_side")
DEFAULT_SOURCE_NAMES = ("all",)


def _parse_csv(value: str | list[str] | tuple[str, ...], default: tuple[str, ...] = ()) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return list(default)
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_float_csv(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(part.strip()) for part in str(value).split(",") if part.strip()]


def _safe_numeric(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce")


def _top_mask(score: pd.Series, frac: float) -> pd.Series:
    mask = pd.Series(False, index=score.index)
    idx = _rank_top_indices(score, frac)
    if len(idx):
        mask.iloc[idx] = True
    return mask


def _mfe_mae(metrics: pd.DataFrame) -> pd.Series:
    ratio = metrics["mfe_norm"] / metrics["mae_norm"].clip(lower=0.25)
    return ratio.replace([np.inf, -np.inf], np.nan).clip(upper=10.0)


def _sigmoid_series(values: Any, index: pd.Index) -> pd.Series:
    arr = np.asarray(values, dtype=np.float64)
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(arr, -60.0, 60.0))), index=index).clip(0.0, 1.0)


def _timestamp_rank(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    rank = _safe_numeric(values).groupby(frame["__ts__"], dropna=False).rank(method="average", pct=True)
    return rank.fillna(_safe_numeric(values).rank(method="average", pct=True)).clip(0.0, 1.0)


def _masked_timestamp_rank(frame: pd.DataFrame, values: pd.Series) -> pd.Series:
    raw = _safe_numeric(values).fillna(0.0).clip(0.0, 1.0)
    return (_timestamp_rank(frame, raw) * raw.gt(0.0).astype(float)).clip(0.0, 1.0)


def _target_frame(soft: pd.Series, hard: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_soft": _safe_numeric(soft).fillna(0.0).clip(0.0, 1.0).astype(np.float32),
            "target_hard": pd.Series(hard, index=soft.index).fillna(False).astype(float),
        },
        index=soft.index,
    )


def _strict_rounda_targets_for_gap(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    base_targets: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Local copy to avoid importing the Round-A script, which imports this diagnostic."""
    idx = metrics.index
    u = _safe_numeric(metrics["u_policy_net"]).fillna(-0.02)
    mae = _safe_numeric(metrics["mae_norm"]).fillna(10.0)
    mfe = _safe_numeric(metrics["mfe_norm"]).fillna(0.0)
    barrier = _safe_numeric(metrics["barrier"]).fillna(1.0)
    bars = _safe_numeric(metrics["bars_to_mfe"]).fillna(24.0)
    timeout = _safe_numeric(metrics["is_timeout"].astype(float)).fillna(1.0).clip(0.0, 1.0)
    mfe_mae = _mfe_mae(metrics).fillna(0.0)

    utility = _sigmoid_series((u - 0.0010) / 0.0060, idx)
    margin_utility = _sigmoid_series((u - 0.0030) / 0.0060, idx)
    low_mae = _sigmoid_series((0.75 - mae) / 0.18, idx)
    low_barrier = _sigmoid_series((0.024 - barrier) / 0.0045, idx)
    efficient = _sigmoid_series((mfe_mae - 1.45) / 0.30, idx)
    fast = _sigmoid_series((7.0 - bars) / 2.5, idx)
    enough_mfe = _sigmoid_series((mfe - 1.15) / 0.25, idx)
    no_timeout = (1.0 - timeout).clip(0.0, 1.0)
    clean_gate = (low_mae * low_barrier * efficient * no_timeout).clip(0.0, 1.0)
    fast_clean_gate = (low_mae * low_barrier * efficient * fast * enough_mfe * no_timeout).clip(0.0, 1.0)

    dirty_penalty = (
        0.35 * _sigmoid_series((mae - 1.00) / 0.20, idx)
        + 0.25 * timeout
        + 0.20 * _sigmoid_series((barrier - 0.025) / 0.004, idx)
        + 0.20 * _sigmoid_series((1.20 - mfe_mae) / 0.25, idx)
    ).clip(0.0, 1.0)

    s3 = base_targets["S3_path_quality"]["target_soft"]
    s8 = base_targets["S8_timestamp_rank_path"]["target_soft"]
    s3_clean = (s3 * utility * clean_gate).clip(0.0, 1.0)
    s8_clean = (s8 * utility * clean_gate).clip(0.0, 1.0)
    s8_clean_rank = _timestamp_rank(frame, s8_clean)
    contrast_raw = (margin_utility * (0.55 * s3 + 0.45 * s8) * clean_gate * (1.0 - dirty_penalty)).clip(0.0, 1.0)
    contrast_rank = _timestamp_rank(frame, contrast_raw)
    fast_clean = (margin_utility * fast_clean_gate).clip(0.0, 1.0)
    fast_clean_rank = _timestamp_rank(frame, fast_clean)

    clean_hard = (
        (u > 0.0010)
        & (mae <= 0.75)
        & (barrier <= 0.025)
        & (mfe_mae >= 1.35)
        & (timeout <= 0.0)
    )
    fast_hard = clean_hard & (bars <= 7.0) & (mfe >= 1.15)
    net_floor = ((u - 0.0010) / 0.0120).clip(0.0, 1.0)
    net_margin_floor = ((u - 0.0030) / 0.0120).clip(0.0, 1.0)
    mae_floor = ((0.95 - mae) / 0.95).clip(0.0, 1.0)
    barrier_floor = ((0.027 - barrier) / 0.027).clip(0.0, 1.0)
    efficiency_floor = ((mfe_mae - 1.05) / 2.00).clip(0.0, 1.0)
    fast_floor = ((9.0 - bars) / 9.0).clip(0.0, 1.0)
    economic_core = (
        (u > 0.0010)
        & (mae <= 0.95)
        & (barrier <= 0.027)
        & (mfe_mae >= 1.05)
        & (timeout <= 0.0)
    )
    economic_core_gate = economic_core.astype(float)
    net_clean_gate = (economic_core_gate * net_floor * mae_floor * barrier_floor * efficiency_floor).clip(0.0, 1.0)
    s3_net_floor = (s3 * net_clean_gate).clip(0.0, 1.0)
    s8_net_floor = (s8 * net_clean_gate).clip(0.0, 1.0)
    s8_net_floor_rank = _masked_timestamp_rank(frame, s8_net_floor)
    direct_clean = (net_margin_floor * net_clean_gate).clip(0.0, 1.0)
    direct_clean_rank = _masked_timestamp_rank(frame, direct_clean)
    fast_net = (direct_clean * fast_floor * enough_mfe).clip(0.0, 1.0)
    fast_net_rank = _masked_timestamp_rank(frame, fast_net)
    return {
        "S120_s3_clean_utility_veto": _target_frame(s3_clean, clean_hard & (s3_clean >= 0.35)),
        "S121_s8_clean_rank_veto": _target_frame((0.45 * s8_clean + 0.55 * s8_clean_rank).clip(0.0, 1.0), clean_hard),
        "S122_clean_dirty_contrast_rank": _target_frame((0.45 * contrast_raw + 0.55 * contrast_rank).clip(0.0, 1.0), clean_hard & (contrast_rank >= 0.85)),
        "S123_fast_clean_path_rank": _target_frame((0.40 * fast_clean + 0.60 * fast_clean_rank).clip(0.0, 1.0), fast_hard & (fast_clean_rank >= 0.85)),
        "S124_s3_net_floor_veto": _target_frame(s3_net_floor, clean_hard & (s3_net_floor > 0.0)),
        "S125_s8_net_floor_rank_veto": _target_frame((0.45 * s8_net_floor + 0.55 * s8_net_floor_rank).clip(0.0, 1.0), clean_hard & (s8_net_floor > 0.0)),
        "S126_clean_net_direct_rank": _target_frame((0.45 * direct_clean + 0.55 * direct_clean_rank).clip(0.0, 1.0), clean_hard & (direct_clean > 0.0)),
        "S127_fast_clean_net_rank": _target_frame((0.40 * fast_net + 0.60 * fast_net_rank).clip(0.0, 1.0), fast_hard & (fast_net > 0.0)),
    }


def _feature_family(feature: str) -> str:
    if feature.startswith("prior_xs_state_"):
        return "state_path_prior"
    if feature.startswith("prior_xs_"):
        return "outcome_prior"
    if feature.startswith("event_xs_") or feature.startswith("event_"):
        return "event_confirmation"
    if feature.startswith("ap_"):
        return "adverse_path_composite"
    if feature.startswith("xs_rank_"):
        return "cross_section_rank"
    tokens = {
        "spread": "liquidity_spread",
        "liquidity": "liquidity_spread",
        "barrier": "barrier_distance",
        "distance_to": "barrier_distance",
        "dist_": "barrier_distance",
        "breakout": "breakout_event",
        "shock": "impulse_event",
        "impulse": "impulse_event",
        "pullback": "pullback_location",
        "loc_": "pullback_location",
        "wick": "exhaustion_reversal",
        "body": "exhaustion_reversal",
        "rejection": "exhaustion_reversal",
        "climax": "exhaustion_reversal",
        "adx": "trend_quality",
        "trend": "trend_quality",
        "oi": "open_interest",
        "volume": "volume_flow",
        "entropy": "chop_compression",
        "compression": "chop_compression",
    }
    for token, family in tokens.items():
        if token in feature:
            return family
    return "other"


def _bucket_key(frame: pd.DataFrame, metrics: pd.DataFrame, mode: str) -> pd.Series:
    ts = pd.to_datetime(frame["__ts__"], errors="coerce")
    side = np.where(_safe_numeric(metrics["side"]).to_numpy(dtype=np.float64, copy=False) < 0.0, "S", "L")
    side_ser = pd.Series(side, index=frame.index, dtype=object)
    ts_key = ts.dt.strftime("%Y-%m-%dT%H:%M:%S").fillna("NA_TS")
    day_key = ts.dt.strftime("%Y-%m-%d").fillna("NA_DAY")
    if mode == "timestamp":
        return ts_key
    if mode == "timestamp_side":
        return ts_key + "|" + side_ser
    if mode == "day_side":
        return day_key + "|" + side_ser
    if mode == "regime_side":
        regime_cols = [
            col
            for col in (
                "__regime_vol_12h__",
                "__regime_vol_48h__",
                "__regime_volume_12h__",
                "__regime_volume_48h__",
                "__regime_trend_12h__",
                "__regime_trend_48h__",
            )
            if col in frame.columns
        ]
        if not regime_cols:
            return side_ser
        parts = [side_ser]
        for col in regime_cols:
            parts.append(frame[col].astype("string").fillna("NA"))
        out = parts[0].astype(str)
        for part in parts[1:]:
            out = out + "|" + part.astype(str)
        return out
    if mode == "regime_day_side":
        return day_key + "|" + _bucket_key(frame, metrics, "regime_side")
    raise ValueError(f"Unknown match mode: {mode}")


def _auc_clean_high(score: pd.Series, clean_mask: pd.Series) -> float:
    mask = score.notna() & clean_mask.notna()
    values = score[mask]
    labels = clean_mask[mask].astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos <= 0 or n_neg <= 0:
        return float("nan")
    ranks = values.rank(method="average")
    rank_sum_pos = float(ranks[labels].sum())
    return (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)


def _rank_within_bucket(values: pd.Series, bucket: pd.Series) -> pd.Series:
    return _safe_numeric(values).groupby(bucket, dropna=False).rank(method="average", pct=True)


def _build_frame(
    *,
    labels_path: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    include_adverse_path_composites: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = _load_labels(labels_path)
    selected_features = _read_feature_list(feature_list_csv, max_features=max_feature_store_features)
    if include_event_confirmation_features:
        selected_features = list(dict.fromkeys(list(selected_features) + list(event_feature_store_features)))
    feature_matrix, feature_store_report = _load_feature_store_columns(
        frame,
        feature_dir=feature_dir,
        selected_features=selected_features,
    )
    if not feature_matrix.empty:
        frame = pd.concat(
            [
                frame.drop(columns=[col for col in feature_matrix.columns if col in frame.columns]),
                feature_matrix.astype(np.float32, copy=False),
            ],
            axis=1,
        ).copy()

    metrics = _path_metrics(frame)
    reports: dict[str, Any] = {"feature_store": feature_store_report}

    reports["causal_outcome_priors"] = {"enabled": False}
    if include_causal_outcome_priors:
        prior_features, reports["causal_outcome_priors"] = _causal_outcome_prior_features(
            frame,
            metrics,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        frame = pd.concat([frame, prior_features.astype(np.float32, copy=False)], axis=1).copy()

    reports["causal_state_path_priors"] = {"enabled": False}
    if include_causal_state_path_priors:
        state_prior_features, reports["causal_state_path_priors"] = _causal_state_path_prior_features(
            frame,
            metrics,
            state_features=state_path_prior_features,
            windows_days=prior_windows_days,
            embargo_hours=prior_embargo_hours,
        )
        frame = pd.concat([frame, state_prior_features.astype(np.float32, copy=False)], axis=1).copy()

    reports["event_confirmation_features"] = {"enabled": False}
    if include_event_confirmation_features:
        event_features, reports["event_confirmation_features"] = _event_confirmation_features(
            frame,
            event_features=event_feature_store_features,
        )
        frame = pd.concat([frame, event_features.astype(np.float32, copy=False)], axis=1).copy()

    reports["adverse_path_composites"] = {"enabled": False}
    if include_adverse_path_composites:
        ap_features, reports["adverse_path_composites"] = _adverse_path_composite_features(frame)
        frame = pd.concat([frame, ap_features.astype(np.float32, copy=False)], axis=1).copy()

    return frame, metrics, reports


def _eligible_masks(
    *,
    bucket: pd.Series,
    missed_clean_mask: pd.Series,
    dirty_false_positive_mask: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, int]:
    candidate = missed_clean_mask | dirty_false_positive_mask
    summary = pd.DataFrame(
        {
            "bucket": bucket,
            "missed_clean": missed_clean_mask.astype(bool),
            "dirty_false_positive": dirty_false_positive_mask.astype(bool),
        }
    )
    bucket_counts = summary.loc[candidate].groupby("bucket", dropna=False).agg(
        missed_clean=("missed_clean", "sum"),
        dirty_false_positive=("dirty_false_positive", "sum"),
    )
    eligible_buckets = bucket_counts[
        bucket_counts["missed_clean"].gt(0) & bucket_counts["dirty_false_positive"].gt(0)
    ].index
    eligible_bucket_mask = bucket.isin(eligible_buckets)
    clean = missed_clean_mask & eligible_bucket_mask
    dirty = dirty_false_positive_mask & eligible_bucket_mask
    eligible_candidate = clean | dirty
    return clean, dirty, eligible_candidate, int(len(eligible_buckets))


def _feature_contrast(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    features: list[str],
    target_train: pd.Series,
    target_valid: pd.Series,
    score: pd.Series,
    bucket: pd.Series,
    clean_mask: pd.Series,
    dirty_mask: pd.Series,
    eligible_candidate_mask: pd.Series,
    proxy_features: list[str],
    min_class_rows: int,
    source: str,
    month: str,
    label_arm: str,
    top_frac: float,
    match_mode: str,
) -> pd.DataFrame:
    clean_count = int(clean_mask.sum())
    dirty_count = int(dirty_mask.sum())
    if clean_count < min_class_rows or dirty_count < min_class_rows:
        return pd.DataFrame()

    proxy_set = set(proxy_features)
    clean_candidate = clean_mask[eligible_candidate_mask].reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for feature in features:
        values = _safe_numeric(valid[feature])
        ranks = _rank_within_bucket(values, bucket)
        clean_ranks = ranks[clean_mask].dropna()
        dirty_ranks = ranks[dirty_mask].dropna()
        if len(clean_ranks) < min_class_rows or len(dirty_ranks) < min_class_rows:
            continue
        candidate_ranks = ranks[eligible_candidate_mask].reset_index(drop=True)
        auc = _auc_clean_high(candidate_ranks, clean_candidate)

        bucket_frame = pd.DataFrame(
            {
                "bucket": bucket[eligible_candidate_mask].reset_index(drop=True),
                "rank": candidate_ranks,
                "clean": clean_candidate,
            }
        ).dropna(subset=["rank"])
        by_bucket = bucket_frame.groupby(["bucket", "clean"], dropna=False)["rank"].mean().unstack()
        if True not in by_bucket.columns or False not in by_bucket.columns:
            continue
        bucket_gap = (by_bucket[True] - by_bucket[False]).dropna()
        if bucket_gap.empty:
            continue

        train_ic = _spearman(train[feature], target_train)
        valid_label_ic = _spearman(valid[feature], target_valid)
        valid_u_ic = _spearman(valid[feature], valid_metrics["u_policy_net"])
        score_ic = _spearman(valid[feature], score)
        best_auc = max(auc, 1.0 - auc) if math.isfinite(auc) else float("nan")
        rows.append(
            {
                "source": source,
                "month": month,
                "label_arm": label_arm,
                "top_frac": float(top_frac),
                "match_mode": match_mode,
                "feature": feature,
                "feature_family": _feature_family(feature),
                "is_proxy_feature": feature in proxy_set,
                "clean_rows": int(len(clean_ranks)),
                "dirty_rows": int(len(dirty_ranks)),
                "matched_buckets_with_feature": int(len(bucket_gap)),
                "clean_rank_mean": float(clean_ranks.mean()),
                "dirty_rank_mean": float(dirty_ranks.mean()),
                "clean_minus_dirty_rank_mean": float(clean_ranks.mean() - dirty_ranks.mean()),
                "bucket_equal_weight_gap_mean": float(bucket_gap.mean()),
                "bucket_equal_weight_gap_median": float(bucket_gap.median()),
                "bucket_gap_positive_rate": float((bucket_gap > 0.0).mean()),
                "auc_clean_high": float(auc),
                "best_auc": float(best_auc),
                "best_direction": "clean_high" if math.isfinite(auc) and auc >= 0.5 else "clean_low",
                "clean_median": float(_safe_numeric(valid.loc[clean_mask, feature]).median()),
                "dirty_median": float(_safe_numeric(valid.loc[dirty_mask, feature]).median()),
                "train_label_ic": train_ic,
                "valid_label_ic": valid_label_ic,
                "valid_utility_ic": valid_u_ic,
                "valid_score_ic": score_ic,
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["abs_bucket_gap"] = out["bucket_equal_weight_gap_mean"].abs()
    out["abs_row_gap"] = out["clean_minus_dirty_rank_mean"].abs()
    out["proxy_rank_penalty"] = np.where(out["is_proxy_feature"], 0, 1)
    return out.sort_values(
        ["best_auc", "abs_bucket_gap", "proxy_rank_penalty"],
        ascending=[False, False, True],
    )


def _summary_row(
    *,
    valid: pd.DataFrame,
    valid_metrics: pd.DataFrame,
    target_valid: pd.DataFrame,
    score: pd.Series,
    oracle_mask: pd.Series,
    proxy_mask: pd.Series,
    missed_clean_mask: pd.Series,
    dirty_false_positive_mask: pd.Series,
    clean_matched_mask: pd.Series,
    dirty_matched_mask: pd.Series,
    eligible_buckets: int,
    source: str,
    month: str,
    label_arm: str,
    top_frac: float,
    match_mode: str,
    proxy_features: list[str],
    top_contrast: pd.DataFrame,
) -> dict[str, Any]:
    proxy_metrics = _selection_metrics(
        frame=valid,
        metrics=valid_metrics,
        target=target_valid,
        score=score,
        arm=f"proxy::{label_arm}",
        selector="label_proxy_oos",
        period=month,
        top_frac=top_frac,
    )
    oracle_metrics = _selection_metrics(
        frame=valid,
        metrics=valid_metrics,
        target=target_valid,
        score=target_valid["target_soft"],
        arm=f"oracle::{label_arm}",
        selector="label_oracle_oos",
        period=month,
        top_frac=top_frac,
    )
    recovered = oracle_mask & proxy_mask
    mfe_mae = _mfe_mae(valid_metrics)
    strict_clean = (
        (valid_metrics["u_policy_net"] > 0.0)
        & (valid_metrics["mae_norm"] <= 0.85)
        & (valid_metrics["barrier"] <= 0.024)
        & (mfe_mae >= 1.35)
        & (~valid_metrics["is_timeout"])
    )
    row = {
        "source": source,
        "month": month,
        "label_arm": label_arm,
        "top_frac": float(top_frac),
        "match_mode": match_mode,
        "valid_rows": int(len(valid)),
        "oracle_top_rows": int(oracle_mask.sum()),
        "proxy_top_rows": int(proxy_mask.sum()),
        "recovered_rows": int(recovered.sum()),
        "oracle_recovery_rate": float(recovered.sum() / oracle_mask.sum()) if int(oracle_mask.sum()) else 0.0,
        "missed_clean_rows": int(missed_clean_mask.sum()),
        "dirty_false_positive_rows": int(dirty_false_positive_mask.sum()),
        "matched_missed_clean_rows": int(clean_matched_mask.sum()),
        "matched_dirty_false_positive_rows": int(dirty_matched_mask.sum()),
        "matched_bucket_count": int(eligible_buckets),
        "score_ic_u": _spearman(score, valid_metrics["u_policy_net"]),
        "score_ic_label": _spearman(score, target_valid["target_soft"]),
        "oracle_mean_return_net": oracle_metrics.get("mean_return_net"),
        "oracle_bad_mae_1r_rate": oracle_metrics.get("bad_mae_1r_rate"),
        "oracle_p90_mae_norm": oracle_metrics.get("p90_mae_norm"),
        "oracle_timeout_rate": oracle_metrics.get("timeout_rate"),
        "oracle_strict_clean_row_rate": oracle_metrics.get("strict_clean_row_rate"),
        "proxy_mean_return_net": proxy_metrics.get("mean_return_net"),
        "proxy_bad_mae_1r_rate": proxy_metrics.get("bad_mae_1r_rate"),
        "proxy_p90_mae_norm": proxy_metrics.get("p90_mae_norm"),
        "proxy_timeout_rate": proxy_metrics.get("timeout_rate"),
        "proxy_strict_clean_row_rate": proxy_metrics.get("strict_clean_row_rate"),
        "proxy_minus_oracle_mean_return_net": (
            proxy_metrics.get("mean_return_net") - oracle_metrics.get("mean_return_net")
            if proxy_metrics.get("mean_return_net") is not None and oracle_metrics.get("mean_return_net") is not None
            else float("nan")
        ),
        "valid_strict_clean_rate": _safe_mean(strict_clean.astype(float)),
        "proxy_features": ",".join(proxy_features),
    }
    if not top_contrast.empty:
        top = top_contrast.iloc[0]
        row.update(
            {
                "top_feature": top.get("feature"),
                "top_feature_family": top.get("feature_family"),
                "top_feature_best_auc": top.get("best_auc"),
                "top_feature_direction": top.get("best_direction"),
                "top_feature_bucket_gap": top.get("bucket_equal_weight_gap_mean"),
                "top_feature_is_proxy_feature": bool(top.get("is_proxy_feature")),
            }
        )
    return row


def _table(frame: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].copy()
    if limit is not None:
        view = view.head(limit)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _write_markdown(
    output_dir: Path,
    summary: pd.DataFrame,
    contrast: pd.DataFrame,
    family_summary: pd.DataFrame,
    source_coverage: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    path = output_dir / "label_matched_clean_dirty_feature_gap.md"
    summary_cols = [
        "source",
        "month",
        "label_arm",
        "top_frac",
        "match_mode",
        "oracle_recovery_rate",
        "missed_clean_rows",
        "dirty_false_positive_rows",
        "matched_missed_clean_rows",
        "matched_dirty_false_positive_rows",
        "matched_bucket_count",
        "oracle_mean_return_net",
        "oracle_bad_mae_1r_rate",
        "oracle_timeout_rate",
        "oracle_strict_clean_row_rate",
        "proxy_mean_return_net",
        "proxy_bad_mae_1r_rate",
        "proxy_p90_mae_norm",
        "proxy_strict_clean_row_rate",
        "proxy_minus_oracle_mean_return_net",
        "top_feature",
        "top_feature_family",
        "top_feature_best_auc",
        "top_feature_direction",
    ]
    contrast_cols = [
        "source",
        "month",
        "label_arm",
        "top_frac",
        "match_mode",
        "feature",
        "feature_family",
        "is_proxy_feature",
        "best_auc",
        "best_direction",
        "bucket_equal_weight_gap_mean",
        "bucket_gap_positive_rate",
        "train_label_ic",
        "valid_label_ic",
        "valid_utility_ic",
    ]
    family_cols = [
        "feature_family",
        "rows",
        "sources",
        "months",
        "labels",
        "mean_best_auc",
        "max_best_auc",
        "mean_abs_bucket_gap",
        "proxy_feature_share",
        "top_features",
    ]
    coverage_cols = [
        "source",
        "period",
        "rows",
        "mean_return_net",
        "bad_mae_1r_rate",
        "timeout_rate",
        "strict_clean_rate",
    ]
    summary_sort_cols = [col for col in ["source", "month", "label_arm", "top_frac", "match_mode"] if col in summary]
    contrast_sort_cols = [col for col in ["best_auc", "abs_bucket_gap"] if col in contrast]
    summary_view = summary.sort_values(summary_sort_cols) if summary_sort_cols else summary
    contrast_view = (
        contrast.sort_values(contrast_sort_cols, ascending=[False, False][: len(contrast_sort_cols)])
        if contrast_sort_cols
        else contrast
    )
    lines = [
        "# Matched Clean-Dirty Label Proxy Feature Gap",
        "",
        "Scope: no model training. Prior-month causal proxies are scored on each OOS month, then missed clean-oracle rows are compared with dirty proxy false positives inside matching buckets.",
        "",
        f"Labels: `{', '.join(manifest['label_arms'])}`",
        f"Months: `{', '.join(manifest['months'])}`",
        f"Top fractions: `{manifest['top_fracs']}`",
        f"Match modes: `{manifest['match_modes']}`",
        f"Sources: `{', '.join(manifest['sources'])}`",
        f"Feature count: `{manifest['feature_count']}`",
        f"Causal outcome priors: `{manifest['include_causal_outcome_priors']}`",
        f"Causal state-path priors: `{manifest['include_causal_state_path_priors']}`",
        f"Event confirmation features: `{manifest['include_event_confirmation_features']}`",
        f"Adverse-path composites: `{manifest['include_adverse_path_composites']}`",
        "",
        "## Source Coverage",
        "",
        _table(source_coverage, coverage_cols, limit=120),
        "",
        "## Confusion Summary",
        "",
        _table(summary_view, summary_cols, limit=200),
        "",
        "## Strongest Matched Separators",
        "",
        _table(
            contrast_view,
            contrast_cols,
            limit=80,
        ),
        "",
        "## Repeated Feature Families",
        "",
        _table(family_summary, family_cols, limit=40),
        "",
        "## Outputs",
        "",
        f"- Summary: `{manifest['outputs']['summary']}`",
        f"- Feature contrast: `{manifest['outputs']['feature_contrast']}`",
        f"- Family summary: `{manifest['outputs']['family_summary']}`",
        f"- Source coverage: `{manifest['outputs']['source_coverage']}`",
        f"- Manifest: `{manifest['outputs']['manifest']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _family_summary(contrast: pd.DataFrame, min_best_auc: float) -> pd.DataFrame:
    if contrast.empty:
        return pd.DataFrame()
    strong = contrast[contrast["best_auc"].ge(float(min_best_auc))].copy()
    if strong.empty:
        strong = contrast.sort_values("best_auc", ascending=False).head(100).copy()
    rows: list[dict[str, Any]] = []
    for family, group in strong.groupby("feature_family", dropna=False, sort=False):
        top_features = (
            group.sort_values("best_auc", ascending=False)["feature"].drop_duplicates().head(8).astype(str).tolist()
        )
        rows.append(
            {
                "feature_family": family,
                "rows": int(len(group)),
                "sources": ",".join(sorted(group["source"].astype(str).unique().tolist()))
                if "source" in group
                else "",
                "months": ",".join(sorted(group["month"].astype(str).unique().tolist())),
                "labels": ",".join(sorted(group["label_arm"].astype(str).unique().tolist())),
                "mean_best_auc": _safe_mean(group["best_auc"]),
                "max_best_auc": _safe_quantile(group["best_auc"], 1.0),
                "mean_abs_bucket_gap": _safe_mean(group["abs_bucket_gap"]),
                "proxy_feature_share": _safe_mean(group["is_proxy_feature"].astype(float)),
                "top_features": ",".join(top_features),
            }
        )
    return pd.DataFrame(rows).sort_values(["max_best_auc", "rows"], ascending=[False, False])


def _source_coverage_rows(
    *,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    sources: dict[str, pd.Series],
    months: list[str],
) -> pd.DataFrame:
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    mfe_mae = _mfe_mae(metrics)
    bad_mae_1r = metrics["mae_norm"] >= 1.0
    strict_clean = (
        (metrics["u_policy_net"] > 0.0)
        & (metrics["mae_norm"] <= 0.85)
        & (metrics["barrier"] <= 0.024)
        & (mfe_mae >= 1.35)
        & (~metrics["is_timeout"])
    )
    periods = ["all", *months]
    rows: list[dict[str, Any]] = []
    for source, source_mask in sources.items():
        source_mask = source_mask.fillna(False).astype(bool).reindex(frame.index, fill_value=False)
        for period in periods:
            period_mask = source_mask if period == "all" else source_mask & month_period.eq(period)
            selected = metrics.loc[period_mask]
            rows.append(
                {
                    "source": source,
                    "period": period,
                    "rows": int(period_mask.sum()),
                    "mean_return_net": _safe_mean(selected["u_policy_net"]) if len(selected) else float("nan"),
                    "bad_mae_1r_rate": _safe_mean(bad_mae_1r.loc[period_mask].astype(float))
                    if int(period_mask.sum())
                    else float("nan"),
                    "timeout_rate": _safe_mean(selected["is_timeout"].astype(float))
                    if len(selected)
                    else float("nan"),
                    "strict_clean_rate": _safe_mean(strict_clean.loc[period_mask].astype(float))
                    if int(period_mask.sum())
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def run_diagnostic(
    *,
    labels_path: Path,
    output_dir: Path,
    feature_dir: Path,
    feature_list_csv: Path,
    max_feature_store_features: int | None,
    months: list[str],
    label_arms: list[str],
    top_fracs: list[float],
    match_modes: list[str],
    proxy_top_k: int,
    include_causal_outcome_priors: bool,
    include_causal_state_path_priors: bool,
    include_event_confirmation_features: bool,
    include_adverse_path_composites: bool,
    prior_windows_days: list[float],
    prior_embargo_hours: float,
    state_path_prior_features: list[str],
    event_feature_store_features: list[str],
    require_strict_clean_oracle: bool,
    min_class_rows: int,
    min_train_rows: int,
    min_valid_rows: int,
    strong_auc_threshold: float,
    sources: list[str],
    run_gap_hours: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, metrics, reports = _build_frame(
        labels_path=labels_path,
        feature_dir=feature_dir,
        feature_list_csv=feature_list_csv,
        max_feature_store_features=max_feature_store_features,
        include_causal_outcome_priors=include_causal_outcome_priors,
        include_causal_state_path_priors=include_causal_state_path_priors,
        include_event_confirmation_features=include_event_confirmation_features,
        include_adverse_path_composites=include_adverse_path_composites,
        prior_windows_days=prior_windows_days,
        prior_embargo_hours=prior_embargo_hours,
        state_path_prior_features=state_path_prior_features,
        event_feature_store_features=event_feature_store_features,
    )
    source_context = _source_context(frame)
    overlap = [col for col in source_context.columns if col in frame.columns]
    if overlap:
        frame = frame.drop(columns=overlap)
    frame = pd.concat([frame, source_context.astype(np.float32, copy=False)], axis=1)
    source_masks_all = _build_sources(frame, source_context, run_gap_hours=float(run_gap_hours))
    requested_sources = list(sources) if sources else list(DEFAULT_SOURCE_NAMES)
    unknown_sources = sorted(set(requested_sources).difference(source_masks_all))
    if unknown_sources:
        raise ValueError(f"Unknown sources: {unknown_sources}. Available: {sorted(source_masks_all)}")
    selected_sources = {
        source: source_masks_all[source].fillna(False).astype(bool).reindex(frame.index, fill_value=False)
        for source in requested_sources
    }

    features = _feature_columns(frame)
    base_targets = _make_targets(frame, metrics)
    targets = _label_targets(frame, metrics)
    targets.update(
        _strict_rounda_targets_for_gap(
            frame=frame,
            metrics=metrics,
            base_targets={**base_targets, **targets},
        )
    )
    unknown = sorted(set(label_arms).difference(targets))
    if unknown:
        raise ValueError(f"Unknown label arms: {unknown}")

    path_scores = _path_targets(metrics)
    month_period = frame["__ts__"].dt.to_period("M").astype(str)
    source_coverage = _source_coverage_rows(frame=frame, metrics=metrics, sources=selected_sources, months=months)
    rows: list[dict[str, Any]] = []
    contrasts: list[pd.DataFrame] = []

    for source_name, source_mask in selected_sources.items():
        for month in months:
            train_mask = (month_period < month) & source_mask
            valid_mask = month_period.eq(month) & source_mask
            if int(train_mask.sum()) < int(min_train_rows) or int(valid_mask.sum()) < int(min_valid_rows):
                continue
            train = frame.loc[train_mask].copy()
            valid_source = frame.loc[valid_mask].copy()
            valid = valid_source.reset_index(drop=True)
            valid_metrics = metrics.loc[valid_mask].copy().reset_index(drop=True)
            valid_strict_clean = path_scores["strict_clean"].loc[valid_mask].reset_index(drop=True).gt(0.5)
            valid_dirty = path_scores["dirty"].loc[valid_mask].reset_index(drop=True).gt(0.5)

            for label_arm in label_arms:
                target = targets[label_arm]
                target_train = target.loc[train_mask, "target_soft"]
                target_valid = target.loc[valid_mask].copy().reset_index(drop=True)
                score, score_diag = _score_proxy(
                    train=train,
                    valid=valid_source,
                    features=features,
                    y_train=target_train,
                    proxy_top_k=proxy_top_k,
                )
                score = score.reset_index(drop=True)
                proxy_features = list(score_diag.get("proxy_features", []))

                for top_frac in top_fracs:
                    oracle_mask = _top_mask(target_valid["target_soft"], float(top_frac))
                    proxy_mask = _top_mask(score, float(top_frac))
                    if require_strict_clean_oracle:
                        missed_clean_mask = oracle_mask & ~proxy_mask & valid_strict_clean
                    else:
                        missed_clean_mask = oracle_mask & ~proxy_mask
                    dirty_false_positive_mask = proxy_mask & ~oracle_mask & valid_dirty

                    for match_mode in match_modes:
                        bucket = _bucket_key(valid, valid_metrics, match_mode).reset_index(drop=True)
                        clean_matched, dirty_matched, eligible_candidate, eligible_buckets = _eligible_masks(
                            bucket=bucket,
                            missed_clean_mask=missed_clean_mask,
                            dirty_false_positive_mask=dirty_false_positive_mask,
                        )
                        contrast = _feature_contrast(
                            train=train,
                            valid=valid,
                            valid_metrics=valid_metrics,
                            features=features,
                            target_train=target_train,
                            target_valid=target_valid["target_soft"],
                            score=score,
                            bucket=bucket,
                            clean_mask=clean_matched,
                            dirty_mask=dirty_matched,
                            eligible_candidate_mask=eligible_candidate,
                            proxy_features=proxy_features,
                            min_class_rows=min_class_rows,
                            source=source_name,
                            month=month,
                            label_arm=label_arm,
                            top_frac=float(top_frac),
                            match_mode=match_mode,
                        )
                        if not contrast.empty:
                            contrasts.append(contrast)
                        rows.append(
                            _summary_row(
                                valid=valid,
                                valid_metrics=valid_metrics,
                                target_valid=target_valid,
                                score=score,
                                oracle_mask=oracle_mask,
                                proxy_mask=proxy_mask,
                                missed_clean_mask=missed_clean_mask,
                                dirty_false_positive_mask=dirty_false_positive_mask,
                                clean_matched_mask=clean_matched,
                                dirty_matched_mask=dirty_matched,
                                eligible_buckets=eligible_buckets,
                                source=source_name,
                                month=month,
                                label_arm=label_arm,
                                top_frac=float(top_frac),
                                match_mode=match_mode,
                                proxy_features=proxy_features,
                                top_contrast=contrast,
                            )
                        )

    summary = pd.DataFrame(rows)
    contrast_all = pd.concat(contrasts, ignore_index=True) if contrasts else pd.DataFrame()
    family = _family_summary(contrast_all, strong_auc_threshold)

    paths = {
        "summary": output_dir / "matched_clean_dirty_summary.csv",
        "feature_contrast": output_dir / "matched_clean_dirty_feature_contrast.csv",
        "family_summary": output_dir / "matched_clean_dirty_family_summary.csv",
        "source_coverage": output_dir / "source_coverage.csv",
        "manifest": output_dir / "manifest.json",
    }
    summary.to_csv(paths["summary"], index=False)
    contrast_all.to_csv(paths["feature_contrast"], index=False)
    family.to_csv(paths["family_summary"], index=False)
    source_coverage.to_csv(paths["source_coverage"], index=False)

    manifest = {
        "scope": "matched_clean_oracle_vs_dirty_proxy_false_positive_feature_gap",
        "labels_path": str(labels_path),
        "output_dir": str(output_dir),
        "rows": int(len(frame)),
        "timestamp_min": frame["__ts__"].min(),
        "timestamp_max": frame["__ts__"].max(),
        "symbols": int(frame["__symbol__"].nunique(dropna=True)),
        "feature_dir": str(feature_dir),
        "feature_list_csv": str(feature_list_csv),
        "max_feature_store_features": max_feature_store_features,
        "feature_count": int(len(features)),
        "months": list(months),
        "label_arms": list(label_arms),
        "strict_rounda_label_arms": list(DEFAULT_STRICT_ROUNDA_LABEL_ARMS),
        "top_fracs": [float(v) for v in top_fracs],
        "match_modes": list(match_modes),
        "sources": list(selected_sources),
        "run_gap_hours": float(run_gap_hours),
        "proxy_top_k": int(proxy_top_k),
        "require_strict_clean_oracle": bool(require_strict_clean_oracle),
        "min_class_rows": int(min_class_rows),
        "min_train_rows": int(min_train_rows),
        "min_valid_rows": int(min_valid_rows),
        "strong_auc_threshold": float(strong_auc_threshold),
        "include_causal_outcome_priors": bool(include_causal_outcome_priors),
        "include_causal_state_path_priors": bool(include_causal_state_path_priors),
        "include_event_confirmation_features": bool(include_event_confirmation_features),
        "include_adverse_path_composites": bool(include_adverse_path_composites),
        "prior_windows_days": [float(v) for v in prior_windows_days],
        "prior_embargo_hours": float(prior_embargo_hours),
        "outputs": {key: str(value) for key, value in paths.items()},
        **reports,
    }
    markdown = _write_markdown(output_dir, summary, contrast_all, family, source_coverage, manifest)
    manifest["outputs"]["markdown"] = str(markdown)
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--feature-list-csv", type=Path, default=DEFAULT_FEATURE_LIST_CSV)
    parser.add_argument("--max-feature-store-features", type=int, default=None)
    parser.add_argument("--months", type=lambda value: _parse_csv(value, DEFAULT_MONTHS), default=",".join(DEFAULT_MONTHS))
    parser.add_argument(
        "--label-arms",
        type=lambda value: _parse_csv(value, DEFAULT_LABEL_ARMS),
        default=",".join(DEFAULT_LABEL_ARMS),
    )
    parser.add_argument(
        "--top-fracs",
        type=_parse_float_csv,
        default=",".join(str(v) for v in DEFAULT_TOP_FRACS),
    )
    parser.add_argument(
        "--match-modes",
        type=lambda value: _parse_csv(value, DEFAULT_MATCH_MODES),
        default=",".join(DEFAULT_MATCH_MODES),
    )
    parser.add_argument(
        "--sources",
        type=lambda value: _parse_csv(value, DEFAULT_SOURCE_NAMES),
        default=",".join(DEFAULT_SOURCE_NAMES),
    )
    parser.add_argument("--run-gap-hours", type=float, default=6.0)
    parser.add_argument("--proxy-top-k", type=int, default=DEFAULT_PROXY_TOP_K)
    parser.add_argument("--include-causal-outcome-priors", action="store_true")
    parser.add_argument("--include-causal-state-path-priors", action="store_true")
    parser.add_argument("--include-event-confirmation-features", action="store_true")
    parser.add_argument("--include-adverse-path-composites", action="store_true")
    parser.add_argument(
        "--prior-windows-days",
        type=_parse_float_csv,
        default=",".join(str(v) for v in DEFAULT_PRIOR_WINDOWS_DAYS),
    )
    parser.add_argument("--prior-embargo-hours", type=float, default=24.0)
    parser.add_argument(
        "--state-path-prior-features",
        type=lambda value: _parse_csv(value, DEFAULT_STATE_PATH_PRIOR_FEATURES),
        default=",".join(DEFAULT_STATE_PATH_PRIOR_FEATURES),
    )
    parser.add_argument(
        "--event-feature-store-features",
        type=lambda value: _parse_csv(value, DEFAULT_EVENT_FEATURE_STORE_FEATURES),
        default=",".join(DEFAULT_EVENT_FEATURE_STORE_FEATURES),
    )
    parser.add_argument("--allow-non-strict-oracle", action="store_true")
    parser.add_argument("--min-class-rows", type=int, default=5)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-valid-rows", type=int, default=100)
    parser.add_argument("--strong-auc-threshold", type=float, default=0.65)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_diagnostic(
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        feature_dir=args.feature_dir,
        feature_list_csv=args.feature_list_csv,
        max_feature_store_features=args.max_feature_store_features,
        months=list(args.months),
        label_arms=list(args.label_arms),
        top_fracs=[float(v) for v in args.top_fracs],
        match_modes=list(args.match_modes),
        proxy_top_k=int(args.proxy_top_k),
        include_causal_outcome_priors=bool(args.include_causal_outcome_priors),
        include_causal_state_path_priors=bool(args.include_causal_state_path_priors),
        include_event_confirmation_features=bool(args.include_event_confirmation_features),
        include_adverse_path_composites=bool(args.include_adverse_path_composites),
        prior_windows_days=[float(v) for v in args.prior_windows_days],
        prior_embargo_hours=float(args.prior_embargo_hours),
        state_path_prior_features=list(args.state_path_prior_features),
        event_feature_store_features=list(args.event_feature_store_features),
        require_strict_clean_oracle=not bool(args.allow_non_strict_oracle),
        min_class_rows=int(args.min_class_rows),
        min_train_rows=int(args.min_train_rows),
        min_valid_rows=int(args.min_valid_rows),
        strong_auc_threshold=float(args.strong_auc_threshold),
        sources=list(args.sources),
        run_gap_hours=float(args.run_gap_hours),
    )
    print(
        json.dumps(
            _json_safe(
                {
                    key: value
                    for key, value in manifest.items()
                    if key not in {
                        "feature_store",
                        "causal_outcome_priors",
                        "causal_state_path_priors",
                        "event_confirmation_features",
                        "adverse_path_composites",
                    }
                }
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
